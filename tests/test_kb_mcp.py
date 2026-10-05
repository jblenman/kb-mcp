"""Unit tests for kb-mcp (stdlib unittest; Python 3.8+).

    python3 tests/test_kb_mcp.py             # or: python3 -m unittest discover tests -v
    /usr/bin/python3 tests/test_kb_mcp.py    # e.g. the macOS system Python 3.9

Builds a temporary index over a small fixture corpus; no network (a fake embedder, or a stub
Ollama on 127.0.0.1, stands in for the embedder). Also runs the CLI and the stdio server as
subprocesses so the script entry points are exercised too.
"""

import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_DIR = HERE.parent / "kb_mcp"
sys.path.insert(0, str(PACKAGE_DIR))

import kb as fk  # noqa: E402
import mcp_server  # noqa: E402

FIXTURE_FILES = {
    "notes/alpha.md": """# Alpha Tool Guide

Intro paragraph about the alpha tool and how the team uses it.

## Installation

Install alpha with the package manager. The installer writes a config file under the home
directory and registers a background service that pulls updates nightly.

## Troubleshooting

### Error 403 on fetch

When alpha answers forbidden, the site has blocked the address. Wait for the back-off and never
retry early. The zebra flag disables the retry loop entirely.

```bash
# this is a shell comment inside a fence, not a heading
alpha fetch --zebra https://example.invalid
```

### Slow startup

Startup is slow when the cache is cold. Warm it with alpha warm.
""",
    "notes/beta.md": """# Beta Service

Beta is the team's message queue. Messages carry a prefix such as task or result.

## Rate limits

The scheduler starts at most six runs per hour and one task at a time; a thirty minute timeout
applies to each run. Claims auto prune after twenty four hours.
""",
    "notes/plain.txt": """Shopping list for the workshop weekend: sandpaper, wood glue, a new blade for the jigsaw,
and the walnut oil finish that the maker space recommended last spring.
""",
    "memory/fact_gamma.md": """---
name: fact-gamma
description: "Gamma rule: never click Update in the Ollama tray on the desktop machine"
metadata:
  type: feedback
---

Clicking Update in the Ollama tray overwrites the custom GPU libraries on the desktop machine.
Start the server from a shell instead and leave the tray alone.
""",
}


def make_long_doc(words_per_para=120, paras=12):
    parts = ["# Long Document", "", "## Big Section", ""]
    for i in range(paras):
        parts.append(" ".join("word%d" % (i * 1000 + j) for j in range(words_per_para)))
        parts.append("")
    return "\n".join(parts)


class FakeEmbedder(object):
    """Deterministic bag-of-words embedding: texts sharing words are close. 32 dims."""

    enabled = True
    model = "fake"
    url = "fake://embedder"
    doc_prefix = ""
    query_prefix = ""
    batch_size = 8
    failed_texts = 0

    def __init__(self, available=True):
        self.ok = available
        self.calls = 0

    @property
    def signature(self):
        return "fake|"

    def available(self, force=False):
        return (True, "fake embedder") if self.ok else (False, "fake embedder down")

    def _vec(self, text):
        vec = [0.0] * 32
        for tok in fk.tokenize(text):
            vec[hash(tok) % 32] += 1.0
        return fk.normalize(vec)

    def embed_documents(self, texts):
        if not self.ok:
            raise fk.EmbedError("down")
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        if not self.ok:
            raise fk.EmbedError("down")
        return self._vec(text)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kbmcp-test-"))
        self.root = self.tmp / "corpus"
        for rel, text in FIXTURE_FILES.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (self.root / "notes" / "long.md").write_text(make_long_doc(), encoding="utf-8")
        (self.root / "notes" / "private").mkdir()
        (self.root / "notes" / "private" / "secret.md").write_text("# Secret\n\nzebra zebra private note\n", encoding="utf-8")
        self.cfg = fk.deep_merge(json.loads(json.dumps(fk.BUILTIN_CONFIG)), {
            "index_path": str(self.tmp / "index.sqlite"),
            "log_file": None,
            "corpora": [
                {"name": "notes", "path": str(self.root / "notes"), "include": ["**/*.md", "**/*.txt"], "exclude": ["private/**"]},
                {"name": "memory", "path": str(self.root / "memory"), "include": ["*.md"]},
                {"name": "everything", "path": str(self.root), "include": ["**/*.md"], "enabled": False},
            ],
            "embedder": {"enabled": True, "url": "http://127.0.0.1:9", "probe_timeout_s": 0.2},
        })
        self.cfg["_paths"] = {}

    def tearDown(self):
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def store(self, backend="auto"):
        return fk.Store(Path(self.cfg["index_path"]), backend)

    def lock_path(self):
        return Path(self.cfg["index_path"]).with_suffix(".lock")

    def write_cfg_file(self, cfg=None, name="config.json"):
        cfg_path = self.tmp / name
        cfg_path.write_text(json.dumps({k: v for k, v in (cfg or self.cfg).items() if k != "_paths"}), encoding="utf-8")
        return cfg_path


class OllamaStub(object):
    """Stand-in for Ollama's /api/tags and /api/embed on 127.0.0.1. Each /api/embed call waits
    `delay` seconds, or until close() when delay is None."""

    def __init__(self, delay=0.05):
        stub = self
        self.delay = delay
        self.embed_calls = 0
        self.released = threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:  # the client is gone (a killed server)
                    pass

            def do_GET(self):
                self._send({"models": [{"name": "nomic-embed-text:latest"}]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                stub.embed_calls += 1
                if stub.delay is None:
                    stub.released.wait(60)
                else:
                    time.sleep(stub.delay)
                self._send({"embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in req.get("input") or []]})

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.block_on_close = False
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def wait_calls(self, n, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.embed_calls >= n:
                return True
            time.sleep(0.01)
        return False

    def close(self):
        self.released.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class ChunkingTests(unittest.TestCase):
    def test_headings_fences_and_line_ranges(self):
        text = FIXTURE_FILES["notes/alpha.md"]
        meta, sections = fk.split_sections(text)
        paths = [s.heading_path for s in sections if s.level]
        self.assertIn("Alpha Tool Guide > Troubleshooting > Error 403 on fetch", paths)
        self.assertNotIn("this is a shell comment inside a fence, not a heading", " ".join(paths))
        lines = text.splitlines()
        for s in sections:
            body = "\n".join(lines[s.line_start - 1:s.line_end]).strip()
            self.assertEqual(body, s.text, "line range must reproduce the section text: %s" % s.heading_path)

    def test_chunks_keep_exact_line_ranges(self):
        text = FIXTURE_FILES["notes/alpha.md"]
        _meta, chunks = fk.chunk_markdown(text, min_words=10, target_words=60, max_words=800)
        lines = text.splitlines()
        self.assertGreaterEqual(len(chunks), 2)
        for ch in chunks:
            self.assertEqual("\n".join(lines[ch.line_start - 1:ch.line_end]).strip(), ch.body)

    def test_small_sections_merge_and_long_sections_split(self):
        _meta, merged = fk.chunk_markdown(FIXTURE_FILES["notes/alpha.md"], min_words=60, target_words=500, max_words=800)
        self.assertEqual(len(merged), 1, "the small sections should merge into one chunk")
        _meta, pieces = fk.chunk_markdown(make_long_doc(), min_words=60, target_words=500, max_words=300)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(p.words <= 300 for p in pieces))
        self.assertTrue(all(p.heading_path == "Long Document > Big Section" for p in pieces))

    def test_frontmatter_description_becomes_heading(self):
        meta, chunks = fk.chunk_markdown(FIXTURE_FILES["memory/fact_gamma.md"])
        self.assertEqual(meta["name"], "fact-gamma")
        self.assertTrue(chunks[0].heading_path.startswith("Gamma rule"))
        self.assertNotIn("type: feedback", chunks[0].body)

    def test_plain_text_without_headings(self):
        text = FIXTURE_FILES["notes/plain.txt"]
        meta, chunks = fk.chunk_markdown(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading_path, "")
        self.assertIn("walnut oil", chunks[0].body)
        _m, sections = fk.split_sections(text)
        self.assertEqual(fk.file_title(meta, sections, "plain.txt"), "plain")
        # a long text file is split on paragraph boundaries like a long section
        _m, pieces = fk.chunk_markdown(make_long_doc().replace("# Long Document\n\n## Big Section\n\n", ""), max_words=300)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(p.words <= 300 for p in pieces))

    def test_tokenizer_and_query_terms(self):
        self.assertEqual(fk.query_terms("the rate-limit per site"), ["rate", "limit", "per", "site"])
        self.assertIn("limit", fk.tokenize("limits"))
        self.assertEqual(fk.query_terms("the a an"), ["the", "a", "an"])  # falls back to all tokens

    def test_deep_merge(self):
        base = {"a": {"x": 1, "y": [1, 2]}, "b": 2}
        out = fk.deep_merge(base, {"a": {"y": [9]}, "c": 3})
        self.assertEqual(out, {"a": {"x": 1, "y": [9]}, "b": 2, "c": 3})

    def test_rrf_order(self):
        lanes = [[(1, 0.0), (2, 0.0), (3, 0.0)], [(2, 0.0), (4, 0.0)]]
        fused = fk.rrf_fuse(lanes, k=20)
        self.assertEqual(fused[0][0], 2, "a chunk ranked in both lanes beats the single-lane #1")
        self.assertEqual(fused[0][2], [2, 1])
        self.assertEqual(fused[1][0], 1)
        weighted = fk.rrf_fuse(lanes, k=20, weights=[1.0, 0.0])
        self.assertEqual([cid for cid, _s, _r in weighted][:3], [1, 2, 3])


class ConfigTests(FixtureCase):
    """Resolution order $KB_MCP_CONFIG -> user file -> example file -> built-in defaults; corpus forms."""

    def setUp(self):
        super(ConfigTests, self).setUp()
        self._env = {k: os.environ.get(k) for k in ("KB_MCP_CONFIG", "KB_MCP_INDEX")}
        for k in self._env:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super(ConfigTests, self).tearDown()

    def test_resolution_order(self):
        user = self.tmp / "user.json"
        example = self.tmp / "example.json"
        missing = self.tmp / "nope.json"
        cfg = fk.load_config(user_path=missing, example_path=missing)
        self.assertEqual(cfg["_paths"]["source"], "builtin")
        self.assertEqual(cfg["corpora"], [])
        self.assertEqual(cfg["embedder"]["url"], "http://127.0.0.1:11434")
        example.write_text(json.dumps({"corpora": [{"name": "ex", "path": str(self.root)}], "_comment": "x"}), encoding="utf-8")
        cfg = fk.load_config(user_path=missing, example_path=example)
        self.assertEqual(cfg["_paths"]["source"], "example")
        self.assertEqual(cfg["corpora"][0]["name"], "ex")
        self.assertNotIn("_comment", cfg, "underscore keys are comments")
        self.assertEqual(cfg["embedder"]["model"], "nomic-embed-text", "defaults fill what the file leaves out")
        user.write_text(json.dumps({"corpora": [{"name": "mine", "path": str(self.root)}], "embedder": {"enabled": False}}), encoding="utf-8")
        cfg = fk.load_config(user_path=user, example_path=example)
        self.assertEqual(cfg["_paths"]["source"], "user", "the user file wins over the example")
        self.assertEqual([c["name"] for c in cfg["corpora"]], ["mine"], "only the first file found is merged")
        self.assertFalse(cfg["embedder"]["enabled"])
        self.assertEqual(cfg["embedder"]["url"], "http://127.0.0.1:11434")
        env_file = self.tmp / "env.json"
        env_file.write_text(json.dumps({"corpora": []}), encoding="utf-8")
        os.environ["KB_MCP_CONFIG"] = str(env_file)
        os.environ["KB_MCP_INDEX"] = str(self.tmp / "elsewhere.sqlite")
        cfg = fk.load_config(user_path=user, example_path=example)
        self.assertEqual(cfg["_paths"]["source"], "env")
        self.assertEqual(cfg["index_path"], str(self.tmp / "elsewhere.sqlite"))

    def test_shipped_example_is_valid(self):
        cfg = fk.load_config(user_path=self.tmp / "nope.json")
        self.assertEqual(cfg["_paths"]["source"], "example")
        self.assertEqual(Path(cfg["_paths"]["loaded"]).name, "config.example.json")
        corpora = fk.resolve_corpora(cfg)
        self.assertEqual([c.name for c in corpora], ["notes", "vault", "claude-memory"])
        self.assertTrue(corpora[0].enabled and not corpora[1].enabled and not corpora[2].enabled)
        self.assertEqual(cfg["index_path"], "~/.claude/kb-mcp/index.sqlite")

    def test_corpus_forms(self):
        single = self.root / "notes" / "alpha.md"
        cfg = dict(self.cfg, corpora=[
            {"name": "one-file", "path": str(single)},
            {"path": str(self.root / "memory")},  # name defaults to the folder name
            {"name": "bad name", "path": str(self.root)},
            {"name": "dup", "path": str(self.root)}, {"name": "dup", "path": str(self.root / "notes")},
            {"name": "legacy", "root": str(self.root / "memory")},
        ])
        corpora = {c.name: c for c in fk.resolve_corpora(cfg)}
        self.assertEqual(sorted(corpora), ["dup", "legacy", "memory", "one-file"])
        self.assertEqual(corpora["one-file"].root, single.parent, "a single file becomes a one-file corpus")
        self.assertEqual(corpora["one-file"].include, ["alpha.md"])
        self.assertEqual(corpora["memory"].include, fk.BUILTIN_CONFIG["default_include"])
        self.assertEqual(corpora["dup"].root, self.root, "the first definition of a duplicate wins")
        self.assertEqual(corpora["legacy"].root, self.root / "memory", "root is accepted as an alias of path")
        files = fk.iter_corpus_files(corpora["one-file"], [], 10 ** 6)
        self.assertEqual([rel for _p, rel in files], ["alpha.md"])
        # an object keyed by name is accepted too; null drops an entry
        as_dict = dict(self.cfg, corpora={"a": {"path": str(self.root / "notes")}, "b": None})
        self.assertEqual([c.name for c in fk.resolve_corpora(as_dict)], ["a"])

    def test_embedder_prefix_defaults(self):
        emb = fk.Embedder({"embedder": {"model": "nomic-embed-text"}})
        self.assertEqual((emb.doc_prefix, emb.query_prefix), ("search_document: ", "search_query: "))
        emb = fk.Embedder({"embedder": {"model": "mxbai-embed-large"}})
        self.assertEqual((emb.doc_prefix, emb.query_prefix), ("", ""))
        emb = fk.Embedder({"embedder": {"model": "nomic-embed-text", "doc_prefix": "", "query_prefix": "q: "}})
        self.assertEqual((emb.doc_prefix, emb.query_prefix), ("", "q: "), "explicit values win, even empty ones")

    def test_init_user_config(self):
        target = self.tmp / "cfgdir" / "config.json"
        cfg = fk.load_config(user_path=target)
        path, created = fk.init_user_config(cfg)
        self.assertTrue(created)
        self.assertEqual(path, target)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["index_path"], "~/.claude/kb-mcp/index.sqlite")
        self.assertEqual(fk.init_user_config(cfg), (target, False))


class IndexAndSearchTests(FixtureCase):
    def test_full_then_incremental(self):
        store = self.store()
        emb = FakeEmbedder()
        r1 = fk.index(self.cfg, store, emb)
        self.assertEqual(r1["files_indexed"], 5)
        self.assertEqual(r1["files_unchanged"], 0)
        self.assertGreater(r1["chunks_written"], 5)
        self.assertEqual(r1["embedding_mode"], "ollama")
        self.assertEqual(r1["embeddings_pending"], 0)
        self.assertNotIn("private", json.dumps([r["rel"] for r in store.con.execute("SELECT rel FROM files")]))
        n_calls = emb.calls

        r2 = fk.index(self.cfg, store, emb)
        self.assertEqual(r2["files_indexed"], 0)
        self.assertEqual(r2["files_unchanged"], 5)
        self.assertEqual(emb.calls, n_calls, "no embedding calls when nothing changed")

        # modify one file: only it is re-chunked; unchanged chunks keep their embeddings
        alpha = self.root / "notes" / "alpha.md"
        alpha.write_text(FIXTURE_FILES["notes/alpha.md"] + "\n## New Section\n\nFresh text about quokkas.\n", encoding="utf-8")
        os.utime(str(alpha), (time.time() + 2, time.time() + 2))
        r3 = fk.index(self.cfg, store, emb)
        self.assertEqual(r3["files_indexed"], 1)
        self.assertEqual(r3["files_unchanged"], 4)
        self.assertEqual(r3["embeddings_pending"], 0)
        hits = fk.search("quokkas", self.cfg, store, emb, k=3)["hits"]
        self.assertTrue(hits and hits[0]["rel"] == "alpha.md")

        # delete a file: purged
        (self.root / "notes" / "beta.md").unlink()
        r4 = fk.index(self.cfg, store, emb)
        self.assertEqual(r4["files_removed"], 1)
        self.assertEqual(fk.search("scheduler runs per hour", self.cfg, store, emb, k=3, mode="lexical")["hits"], [])

        # a corpus removed from the config is purged from the index
        cfg2 = dict(self.cfg, corpora=[c for c in self.cfg["corpora"] if c["name"] != "memory"])
        r5 = fk.index(cfg2, store, emb)
        self.assertIn("memory", r5["corpora"])
        self.assertNotIn("memory", store.corpus_stats())
        store.close()

    def test_lexical_search_and_modes(self):
        store = self.store()
        emb = FakeEmbedder()
        fk.index(self.cfg, store, emb)
        res = fk.search("scheduler runs per hour", self.cfg, store, emb, k=3, mode="lexical")
        self.assertEqual(res["mode"], "lexical")
        self.assertEqual(res["hits"][0]["rel"], "beta.md")
        self.assertEqual(res["hits"][0]["heading"], "Beta Service", "a small file is one chunk named after its H1")
        self.assertIn("Rate limits", res["hits"][0]["snippet"])
        self.assertLessEqual(len(res["hits"][0]["snippet"]), 610)

        res = fk.search("Ollama tray Update desktop", self.cfg, store, emb, k=3)
        self.assertEqual(res["mode"], "hybrid")
        self.assertEqual(res["hits"][0]["corpus"], "memory")
        self.assertIsNotNone(res["hits"][0]["vec_rank"])
        self.assertGreater(res["lexical_coverage"], 0.5)
        self.assertLess(res["vec_weight"], 1.0)

        # plain-text files are searchable like any note
        res = fk.search("walnut oil finish", self.cfg, store, emb, k=3, mode="lexical")
        self.assertEqual(res["hits"][0]["rel"], "plain.txt")
        self.assertEqual(res["hits"][0]["title"], "plain")

        # corpus filter
        res = fk.search("Ollama tray Update desktop", self.cfg, store, emb, k=3, corpora=["notes"])
        self.assertTrue(all(h["corpus"] == "notes" for h in res["hits"]))
        store.close()

    def test_embedder_down_degrades_to_lexical(self):
        store = self.store()
        down = FakeEmbedder(available=False)
        r = fk.index(self.cfg, store, down)
        self.assertEqual(r["embedding_mode"], "unavailable")
        self.assertEqual(r["embeddings_pending"], r["chunks_written"])
        res = fk.search("zebra flag", self.cfg, store, down, k=3)
        self.assertEqual(res["mode"], "lexical")
        self.assertIn("no embeddings", res["note"])
        self.assertEqual(res["hits"][0]["rel"], "alpha.md")
        # embedder comes back: the backlog is embedded, search turns hybrid
        up = FakeEmbedder(available=True)
        r = fk.index(self.cfg, store, up)
        self.assertEqual(r["embeddings_pending"], 0)
        self.assertEqual(fk.search("zebra flag", self.cfg, store, up, k=3)["mode"], "hybrid")
        # embedder present but unreachable at query time
        up.ok = False
        res = fk.search("zebra flag", self.cfg, store, up, k=3)
        self.assertEqual(res["mode"], "lexical")
        self.assertIn("down", res["note"])
        store.close()

    def test_embedder_disabled_never_contacts_anything(self):
        cfg = fk.deep_merge(self.cfg, {"embedder": {"enabled": False, "url": "http://127.0.0.1:9"}})
        store = self.store()
        emb = fk.Embedder(cfg)
        r = fk.index(cfg, store, emb)
        self.assertEqual(r["embedding_mode"], "disabled")
        self.assertEqual(r["embeddings_pending"], r["chunks_written"])
        res = fk.search("zebra flag", cfg, store, emb, k=3)
        self.assertEqual(res["mode"], "lexical")
        self.assertEqual(res["note"], "", "no probe, no note: lexical is the configured behaviour")
        self.assertEqual(res["hits"][0]["rel"], "alpha.md")
        st = fk.status(cfg, store, emb, probe=True)
        self.assertFalse(st["embeddings"]["enabled"])
        self.assertIn("disabled", fk.format_status(st))
        store.close()

    def test_python_bm25_backend(self):
        store = self.store(backend="python")
        self.assertEqual(store.backend, "python")
        fk.index(self.cfg, store, FakeEmbedder(available=False))
        res = fk.search("scheduler runs per hour", self.cfg, store, None, k=3)
        self.assertEqual(res["lexical_backend"], "python-bm25")
        self.assertEqual(res["hits"][0]["rel"], "beta.md")
        store.close()
        if store.has_fts5:
            # switching backends rebuilds the lexical index instead of failing
            store2 = self.store(backend="fts5")
            res = fk.search("scheduler runs per hour", self.cfg, store2, None, k=3)
            self.assertEqual(res["hits"][0]["rel"], "beta.md")
            store2.close()

    def test_get_recent_status(self):
        store = self.store()
        fk.index(self.cfg, store, FakeEmbedder())
        out = fk.get_section("notes:alpha.md", "Error 403", self.cfg, store)
        self.assertNotIn("error", out)
        self.assertTrue(out["text"].startswith("### Error 403 on fetch"))
        self.assertIn("zebra", out["text"])
        self.assertNotIn("Slow startup", out["text"], "the next sibling heading ends the section")
        parent = fk.get_section("alpha.md", "Troubleshooting", self.cfg, store)
        self.assertIn("Slow startup", parent["text"], "a parent heading includes its sub-sections")
        whole = fk.get_section(str(self.root / "notes" / "alpha.md"), None, self.cfg, store)
        self.assertEqual(whole["line_start"], 1)
        self.assertIn("error", fk.get_section("notes:alpha.md", "nonexistent heading", self.cfg, store))
        self.assertIn("error", fk.get_section("/etc/hosts", None, self.cfg, store))

        recent = fk.recent_files(self.cfg, store, None, 1, 10)
        self.assertEqual(len(recent), 5)
        self.assertEqual(len(fk.recent_files(self.cfg, store, "memory", 1, 10)), 1)

        st = fk.status(self.cfg, store, None, probe=False)
        self.assertEqual(st["files"], 5)
        self.assertEqual(st["embeddings"]["coverage"], 1.0)
        self.assertTrue(any(c["name"] == "everything" and not c["available"] for c in st["corpora"]))
        self.assertEqual(st["hint"], "", "a working config needs no hint")
        self.assertTrue(st["cli"].endswith("kb.py"))
        self.assertIn("(disabled)", fk.format_status(st))

        # hints when nothing useful is configured
        empty = dict(self.cfg, corpora=[], _paths={"user": "U", "source": "builtin"})
        self.assertIn("no corpora configured", fk.config_hint(empty, fk.resolve_corpora(empty)))
        gone = dict(self.cfg, corpora=[{"name": "x", "path": str(self.tmp / "missing")}], _paths={"user": "U", "loaded": "L", "source": "user"})
        self.assertIn("no corpus folder exists", fk.config_hint(gone, fk.resolve_corpora(gone)))
        ex = dict(self.cfg, _paths={"user": "U", "loaded": "E", "source": "example"})
        self.assertIn("config.example.json", fk.config_hint(ex, fk.resolve_corpora(ex)))
        store.close()

    def test_vector_math(self):
        vec = [3.0, 4.0]
        packed = fk.pack_vector(fk.normalize(vec))
        back = fk.unpack_vector(packed)
        self.assertAlmostEqual(back[0], 0.6, places=5)
        self.assertAlmostEqual(back[1], 0.8, places=5)


class McpTests(FixtureCase):
    def setUp(self):
        super(McpTests, self).setUp()
        store = self.store()
        fk.index(self.cfg, store, FakeEmbedder())
        store.close()
        self.server = mcp_server.Server(self.cfg, autoindex=False)
        self.server.embedder = FakeEmbedder()

    def rpc(self, method, params=None, msg_id=1):
        msg = {"jsonrpc": "2.0", "method": method}
        if msg_id is not None:
            msg["id"] = msg_id
        if params is not None:
            msg["params"] = params
        return self.server.handle_message(msg)

    def test_handshake(self):
        resp = self.rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
        self.assertEqual(resp["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(resp["result"]["serverInfo"]["name"], "kb-mcp")
        self.assertIn("tools", resp["result"]["capabilities"])
        self.assertIn("not instructions", resp["result"]["instructions"])
        self.assertIn("notes", resp["result"]["instructions"], "the instructions name the configured corpora")
        resp = self.rpc("initialize", {"protocolVersion": "2025-11-25"})
        self.assertEqual(resp["result"]["protocolVersion"], "2025-11-25", "what Claude Code 2.1.286+ asks for is echoed")
        resp = self.rpc("initialize", {"protocolVersion": "1999-01-01"})
        self.assertEqual(resp["result"]["protocolVersion"], "2025-06-18", "unknown version -> the fallback")
        resp = self.rpc("initialize", {})
        self.assertEqual(resp["result"]["protocolVersion"], "2025-06-18", "no version -> the same fallback")
        self.assertIsNone(self.rpc("notifications/initialized", msg_id=None))
        self.assertEqual(self.rpc("ping")["result"], {})
        self.assertEqual(self.rpc("server/discover")["error"]["code"], -32601, "the probe some clients send first")
        self.assertEqual(self.rpc("bogus")["error"]["code"], -32601)
        self.assertIsNone(self.rpc("notifications/bogus", msg_id=None))

    def test_tools_list_and_call(self):
        tools = self.rpc("tools/list")["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["kb_search", "kb_get", "kb_recent", "kb_reindex", "kb_status"])
        self.assertIn("notes", tools[0]["inputSchema"]["properties"]["corpora"]["items"]["enum"])
        for t in tools:
            self.assertEqual(t["inputSchema"]["type"], "object")

        res = self.rpc("tools/call", {"name": "kb_search", "arguments": {"query": "scheduler runs per hour", "k": 2}})["result"]
        self.assertFalse(res["isError"])
        text = res["content"][0]["text"]
        self.assertIn("notes:beta.md", text)
        self.assertIn("mode=", text)
        self.assertIn("kb_get", text)

        res = self.rpc("tools/call", {"name": "kb_get", "arguments": {"path": "notes:beta.md", "heading": "Rate limits"}})["result"]
        self.assertFalse(res["isError"])
        self.assertIn("six runs per hour", res["content"][0]["text"])

        res = self.rpc("tools/call", {"name": "kb_recent", "arguments": {"days": 1}})["result"]
        self.assertIn("memory:fact_gamma.md", res["content"][0]["text"])

        res = self.rpc("tools/call", {"name": "kb_status", "arguments": {}})["result"]
        self.assertIn("5 files", res["content"][0]["text"])

        res = self.rpc("tools/call", {"name": "kb_reindex", "arguments": {}})["result"]
        self.assertIn("5 unchanged", res["content"][0]["text"])

        res = self.rpc("tools/call", {"name": "kb_search", "arguments": {"query": "x", "corpora": ["nope"]}})["result"]
        self.assertTrue(res["isError"])
        res = self.rpc("tools/call", {"name": "unknown_tool", "arguments": {}})["result"]
        self.assertTrue(res["isError"])

    def test_empty_config_is_explained(self):
        cfg = dict(self.cfg, corpora=[], index_path=str(self.tmp / "empty.sqlite"), _paths={"user": "U", "source": "builtin"})
        server = mcp_server.Server(cfg, autoindex=False)
        try:
            resp = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            self.assertIn("no corpora are configured", resp["result"]["instructions"])
            res = server.call_tool("kb_search", {"query": "anything"})
            self.assertFalse(res["isError"])
            self.assertIn("no corpora configured", res["content"][0]["text"])
        finally:
            server.store.close()


class CliTests(FixtureCase):
    """The script entry points as subprocesses: kb_mcp/kb.py and kb_mcp/mcp_server.py."""

    def run_cli(self, *args):
        env = dict(os.environ)
        env["KB_MCP_CONFIG"] = str(self.write_cfg_file())
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("KB_MCP_INDEX", None)
        return subprocess.run([sys.executable, str(PACKAGE_DIR / "kb.py")] + list(args), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=env, universal_newlines=True, timeout=120)

    def test_cli(self):
        r = self.run_cli("index", "--no-embed", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["files_indexed"], 5)
        r = self.run_cli("search", "zebra flag", "--mode", "lexical", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["hits"][0]["rel"], "alpha.md")
        r = self.run_cli("status", "--no-probe", "--json")
        self.assertEqual(json.loads(r.stdout)["files"], 5)
        r = self.run_cli("index", "--no-embed", "--if-older-than", "60")
        self.assertIn("nothing to do", r.stdout)
        r = self.run_cli("get", "notes:beta.md", "--heading", "Rate limits")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("six runs per hour", r.stdout)
        r = self.run_cli("recent", "--days", "1")
        self.assertIn("notes:alpha.md", r.stdout)
        r = self.run_cli("config")
        self.assertEqual(json.loads(r.stdout)["_paths"]["source"], "env")
        r = subprocess.run([sys.executable, "-m", "kb_mcp", "config"], cwd=str(HERE.parent), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, timeout=60,
                           env=dict(os.environ, KB_MCP_CONFIG=str(self.tmp / "config.json")))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("index_path", r.stdout)

    def test_mcp_server_stdio(self):
        env = dict(os.environ, KB_MCP_CONFIG=str(self.write_cfg_file()), KB_MCP_NO_AUTOINDEX="1")
        env.pop("KB_MCP_INDEX", None)
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "kb_reindex", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "kb_search", "arguments": {"query": "zebra flag", "mode": "lexical"}}},
        ]
        proc = subprocess.run([sys.executable, str(PACKAGE_DIR / "mcp_server.py")], input="\n".join(json.dumps(m) for m in msgs) + "\n",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual([m["id"] for m in lines], [1, 2, 3, 4], "one response per request, none for the notification")
        self.assertEqual(lines[0]["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(len(lines[1]["result"]["tools"]), 5)
        self.assertIn("notes:alpha.md", lines[3]["result"]["content"][0]["text"])
        self.assertEqual(proc.stderr.strip(), "", "nothing below WARNING reaches stderr")


class LockTests(FixtureCase):
    """index.lock: an owner that is gone is taken over at once; a live one is respected."""

    def setUp(self):
        super(LockTests, self).setUp()
        self.procs = []  # kept referenced: on Windows an open handle stops the PID being reused mid-test

    def tearDown(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        super(LockTests, self).tearDown()

    def spawn(self, code):
        proc = subprocess.Popen([sys.executable, "-c", code])
        self.procs.append(proc)
        return proc

    def dead_pid(self):
        proc = self.spawn("pass")
        proc.wait()
        return proc.pid

    def write_lock(self, raw):
        self.lock_path().write_bytes(raw)

    @staticmethod
    def lock_bytes(pid, created=None):
        return json.dumps({"pid": pid, "created": time.time() if created is None else created}).encode("ascii")

    def index_once(self):
        store = self.store()
        try:
            return fk.index(self.cfg, store, None)
        finally:
            store.close()

    def test_pid_alive(self):
        self.assertTrue(fk._pid_alive(os.getpid()))
        self.assertFalse(fk._pid_alive(self.dead_pid()))
        self.assertIsNone(fk._pid_alive(0), "0 and negatives are process groups for os.kill, never checked")
        self.assertIsNone(fk._pid_alive(-1))
        self.assertIsNone(fk._pid_alive(10 ** 30), "a corrupt record cannot crash the index run")

    def test_dead_owner_is_taken_over_at_once(self):
        self.write_lock(self.lock_bytes(self.dead_pid()))
        report = self.index_once()
        self.assertFalse(report["skipped"], report.get("reason"))
        self.assertEqual(report["files_indexed"], 5)
        self.assertFalse(self.lock_path().exists(), "the run releases the lock it took over")
        self.assertEqual(list(self.tmp.glob("*.stale")), [], "no takeover file left behind")

    def test_lock_record_parsing_is_tolerant(self):
        self.assertEqual(fk.IndexLock.parse(b"4242 1700000000.500000"), (4242, 1700000000.5))
        self.assertEqual(fk.IndexLock.parse(self.lock_bytes(4242, 1.5)), (4242, 1.5))
        self.assertEqual(fk.IndexLock.parse(b""), (None, None))
        self.assertEqual(fk.IndexLock.parse(b"{not json"), (None, None))
        self.write_lock(("%d %f" % (self.dead_pid(), time.time())).encode("ascii"))  # the plain-text record form
        self.assertFalse(self.index_once()["skipped"])

    def test_live_owner_is_respected_until_it_exits(self):
        owner = self.spawn("import time; time.sleep(60)")
        self.write_lock(self.lock_bytes(owner.pid))
        report = self.index_once()
        self.assertTrue(report["skipped"])
        self.assertIn("another index run", report["reason"])
        self.assertTrue(self.lock_path().exists())
        owner.kill()
        owner.wait()
        self.assertFalse(self.index_once()["skipped"], "the same lock is taken over once its owner is gone")

    def test_unknown_content_is_respected_until_the_age_limit(self):
        self.write_lock(b"")  # an O_EXCL file whose owner has not written its record yet
        self.assertTrue(self.index_once()["skipped"])
        old = time.time() - 1000
        os.utime(str(self.lock_path()), (old, old))
        self.assertFalse(self.index_once()["skipped"])

    def test_age_limit_still_applies_to_a_live_pid(self):
        self.write_lock(self.lock_bytes(os.getpid()))  # a hung owner, or a reused PID POSIX cannot tell apart
        old = time.time() - 1000
        os.utime(str(self.lock_path()), (old, old))
        self.assertFalse(self.index_once()["skipped"])

    def test_takeover_puts_back_a_lock_taken_in_between(self):
        stale = self.lock_bytes(self.dead_pid())
        fresh = self.lock_bytes(os.getpid())
        self.write_lock(fresh)  # another run took over between our read (stale) and our break
        self.assertTrue(fk.IndexLock(Path(self.cfg["index_path"]))._break(stale))
        self.assertEqual(self.lock_path().read_bytes(), fresh)
        self.assertEqual(list(self.tmp.glob("*.stale")), [])

    def test_release_leaves_a_lock_that_is_not_ours(self):
        lock = fk.IndexLock(Path(self.cfg["index_path"]))
        other = self.lock_bytes(os.getpid(), 1.0)
        with lock:
            self.assertTrue(lock.acquired)
            self.write_lock(other)  # taken over while this run held it (the age limit)
        self.assertEqual(self.lock_path().read_bytes(), other)
        with fk.IndexLock(self.tmp / "second.sqlite") as mine:
            self.assertTrue(mine.acquired)
        self.assertFalse((self.tmp / "second.lock").exists(), "our own lock is released")

    @unittest.skipUnless(os.name == "nt", "start times are checked on Windows only (POSIX hands PIDs out in sequence)")
    def test_reused_pid_is_gone_on_windows(self):
        proc = self.spawn("import time; time.sleep(60)")
        self.assertTrue(fk._pid_alive(proc.pid, time.time()), "lock taken after the process started")
        self.assertFalse(fk._pid_alive(proc.pid, time.time() - 3600), "a process younger than the lock is not its owner")


class StopTests(FixtureCase):
    """index(should_stop=...): commits what it did, releases the lock, purges nothing, stays stale."""

    def test_stop_during_the_walk(self):
        store = self.store()
        fk.index(self.cfg, store, None)
        first_run = store.get_meta("last_index_run")
        (self.root / "notes" / "beta.md").unlink()
        checks = []
        report = fk.index(self.cfg, store, None, should_stop=lambda: checks.append(1) or len(checks) > 2)
        self.assertTrue(report["stopped"])
        self.assertIn("next run", report["reason"])
        self.assertEqual(report["files_removed"], 0, "a partial walk must not purge files it never reached")
        self.assertFalse(self.lock_path().exists())
        self.assertEqual(store.get_meta("last_index_run"), first_run, "a stopped run does not mark the index fresh")
        self.assertEqual(fk.index(self.cfg, store, None)["files_removed"], 1, "the next full run purges it")
        store.close()

    def test_stop_during_embedding(self):
        store = self.store()
        emb = FakeEmbedder()
        emb.batch_size = 1
        report = fk.index(self.cfg, store, emb, should_stop=lambda: emb.calls >= 2)
        self.assertTrue(report["stopped"])
        self.assertEqual(report["files_indexed"], 5, "the lexical phase finished first")
        self.assertEqual(report["embedding_mode"], "stopped")
        self.assertEqual(report["embeddings_new"], 2)
        self.assertGreater(report["embeddings_pending"], 0)
        self.assertFalse(self.lock_path().exists())
        self.assertIsNone(fk.index_age_seconds(store))
        self.assertEqual(fk.index(self.cfg, store, emb)["embeddings_pending"], 0, "the next run embeds the rest")
        store.close()


class ExitTests(FixtureCase):
    """The MCP server's exit path with a background index running (stub Ollama, real server process)."""

    def setUp(self):
        super(ExitTests, self).setUp()
        self.stub = None
        for i in range(30):  # one chunk each: a long embedding phase at batch_size 1
            (self.root / "notes" / ("extra%02d.md" % i)).write_text(
                "# Extra %d\n\n%s\n" % (i, " ".join("w%d_%d" % (i, j) for j in range(80))), encoding="utf-8")

    def tearDown(self):
        if self.stub is not None:
            self.stub.close()
        super(ExitTests, self).tearDown()

    def stub_cfg(self, delay):
        self.stub = OllamaStub(delay)
        cfg = fk.deep_merge(self.cfg, {"log_file": str(self.tmp / "kb-mcp.log"),
                                       "embedder": {"url": self.stub.url, "model": "nomic-embed-text", "batch_size": 1,
                                                    "timeout_s": 90, "probe_timeout_s": 5}})
        cfg["_paths"] = {}
        return cfg

    def start_server(self, cfg):
        env = dict(os.environ, KB_MCP_CONFIG=str(self.write_cfg_file(cfg)))
        for key in ("KB_MCP_NO_AUTOINDEX", "KB_MCP_INDEX"):
            env.pop(key, None)
        proc = subprocess.Popen([sys.executable, str(PACKAGE_DIR / "mcp_server.py")], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        for msg in ({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"}):  # starts the background index
            proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        proc.stdin.flush()
        self.assertEqual(json.loads(proc.stdout.readline().decode("utf-8"))["result"]["protocolVersion"], "2025-11-25")
        return proc

    def test_shutdown_stops_the_background_index(self):
        server = mcp_server.Server(self.stub_cfg(0.05))
        server.autoindex = True
        try:
            self.assertTrue(server.maybe_refresh("test"))
            self.assertTrue(self.stub.wait_calls(2), "the background index reached the embedding phase")
            self.assertTrue(self.lock_path().exists())
            self.assertTrue(server.shutdown(wait_s=10.0))
            self.assertFalse(self.lock_path().exists(), "released by the stopped run")
            report = server.last_refresh_report
            self.assertTrue(report["stopped"])
            self.assertEqual(report["embedding_mode"], "stopped")
            self.assertGreater(report["embeddings_pending"], 0)
            self.assertIsNone(fk.index_age_seconds(server.store), "not marked fresh: the next start finishes the job")
            self.assertFalse(server.maybe_refresh("after shutdown"), "no new run once stopping")
        finally:
            server.store.close()

    def test_stdin_close_mid_index_releases_the_lock(self):
        proc = self.start_server(self.stub_cfg(0.05))
        try:
            self.assertTrue(self.stub.wait_calls(2), "the background index reached the embedding phase")
            self.assertTrue(self.lock_path().exists())
            t = time.time()
            _out, err = proc.communicate(timeout=30)  # closes stdin: EOF
            elapsed = time.time() - t
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual(proc.returncode, 0, err)
        self.assertLess(elapsed, mcp_server.EXIT_WAIT_S + 2.0)
        self.assertFalse(self.lock_path().exists(), "the stopped run released index.lock")
        err = err.decode("utf-8", "replace")
        self.assertNotIn("INFO", err, "INFO stays out of stderr (Claude Code records stderr as [ERROR])")
        self.assertNotIn("ready", err)
        log_text = (self.tmp / "kb-mcp.log").read_text(encoding="utf-8")
        self.assertIn("ready", log_text, "INFO goes to the log file")
        self.assertIn("stopped early", log_text)

    def test_killed_server_leaves_a_lock_the_next_run_takes_over(self):
        proc = self.start_server(self.stub_cfg(None))  # embedding calls hang: nothing can stop in time
        try:
            self.assertTrue(self.stub.wait_calls(1))
            self.assertEqual(fk.IndexLock.parse(self.lock_path().read_bytes())[0], proc.pid)
        finally:
            proc.kill()  # SIGKILL / TerminateProcess: no code in the server runs
            proc.communicate(timeout=30)
        self.assertTrue(self.lock_path().exists(), "a killed server cannot release its lock")
        store = self.store()
        try:
            report = fk.index(self.cfg, store, None)  # what the next session's server does
        finally:
            store.close()
        self.assertFalse(report["skipped"], "a lock naming a dead PID is taken over at once, not after 15 minutes")
        self.assertFalse(self.lock_path().exists())

    @unittest.skipIf(os.name == "nt", "POSIX signals; on Windows a server is killed outright (previous test)")
    def test_claude_code_exit_signals(self):
        # Claude Code ends a stdio server with SIGINT, then SIGTERM ~0.1 s later (then SIGKILL)
        proc = self.start_server(self.stub_cfg(0.4))
        try:
            self.assertTrue(self.stub.wait_calls(1))
            self.assertTrue(self.stub.wait_calls(self.stub.embed_calls + 1))  # a call just started: ~0.4 s left in it
            proc.send_signal(signal.SIGINT)
            time.sleep(0.1)
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)  # must not cut the wait for the index thread short
            proc.communicate(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual(proc.returncode, 0, "the first signal became a normal exit")
        self.assertFalse(self.lock_path().exists(), "the index thread stopped and released the lock before exit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
