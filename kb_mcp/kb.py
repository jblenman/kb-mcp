"""kb-mcp: a local, private search index over folders of Markdown (and text) notes.

Corpora come from a config file (see config.example.json): each one is a folder (or a single
file) plus include/exclude globs. The index is one SQLite file under the home folder,
never inside a notes folder and never in a repository.

Retrieval: BM25 over SQLite FTS5 (a pure-Python BM25 over a postings table when the
interpreter's sqlite3 lacks FTS5), fused by reciprocal rank fusion with cosine similarity
over embeddings from an Ollama endpoint (nomic-embed-text by default, stored as float32
blobs). When the embedder is disabled or unreachable the service answers lexically and
says so; it never fails because the embedder is away.

CLI:
    kb.py index [--full] [--corpus NAME ...] [--no-embed] [--if-older-than MIN] [--json]
    kb.py search QUERY [--corpus NAME ...] [-k N] [--mode auto|lexical|hybrid] [--json]
    kb.py get PATH [--heading TEXT] [--max-chars N] [--json]
    kb.py recent [--corpus NAME] [--days N] [--limit N] [--json]
    kb.py status [--json] [--no-probe]
    kb.py config [--init]

Config resolution: $KB_MCP_CONFIG, else ~/.claude/kb-mcp/config.json, else config.example.json
beside this package; the file found is merged over the built-in defaults. $KB_MCP_INDEX
overrides the index path.

Stdlib only, Python 3.8+.
"""

import argparse
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from array import array
from collections import defaultdict
from datetime import datetime
from operator import mul
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

__version__ = "1.0.0"
SCHEMA_VERSION = 1

HERE = Path(__file__).resolve().parent
EXAMPLE_CONFIG_PATH = HERE.parent / "config.example.json"
USER_CONFIG_PATH = Path.home() / ".claude" / "kb-mcp" / "config.json"

log = logging.getLogger("kb_mcp")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

BUILTIN_CONFIG = {
    "index_path": "~/.claude/kb-mcp/index.sqlite",
    "log_file": "~/.claude/kb-mcp/kb-mcp.log",
    "corpora": [],
    "default_include": ["**/*.md", "**/*.markdown", "**/*.txt"],
    "exclude_globs": ["**/.git/**", "**/node_modules/**", "**/.obsidian/**", "**/.trash/**"],
    "max_file_bytes": 2000000,
    "chunking": {"min_words": 60, "target_words": 500, "max_words": 800},
    "embedder": {
        "enabled": True,
        "url": "http://127.0.0.1:11434",
        "model": "nomic-embed-text",
        "doc_prefix": None,    # None = "search_document: " for nomic-embed-text, "" for other models
        "query_prefix": None,  # None = "search_query: " for nomic-embed-text, "" for other models
        "batch_size": 32,
        "timeout_s": 120,
        "probe_timeout_s": 2,
        "max_chars": 6000,
        "num_ctx": 8192,
    },
    "lexical": {"backend": "auto"},
    "search": {"k": 6, "pool": 40, "vec_pool": 20, "rrf_k": 20, "vec_weight_min": 0.2, "vec_weight_max": 1.0,
               "snippet_chars": 600, "max_per_file": 3},
    "freshness": {"auto_index_minutes": 30},
}


def deep_merge(base: Any, override: Any) -> Any:
    """Objects merge key by key, lists and scalars replace."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for key, value in override.items():
            out[key] = deep_merge(base.get(key), value) if key in base else value
        return out
    return override


def _strip_comments(data: Any) -> Any:
    """Keys starting with "_" are comments in the config files (JSON has no comment syntax)."""
    if isinstance(data, dict):
        return {k: _strip_comments(v) for k, v in data.items() if not str(k).startswith("_")}
    if isinstance(data, list):
        return [_strip_comments(v) for v in data]
    return data


def _read_json(path: Path) -> Dict[str, Any]:
    with open(str(path), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("%s: top level must be an object" % path)
    return _strip_comments(data)


def config_candidates(user_path: Optional[Path] = None, example_path: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """(source label, path) in resolution order: $KB_MCP_CONFIG, the user file, the example file."""
    out: List[Tuple[str, Path]] = []
    env = os.environ.get("KB_MCP_CONFIG")
    if env:
        out.append(("env", Path(expand_user_path(env))))
    out.append(("user", user_path or USER_CONFIG_PATH))
    out.append(("example", example_path or EXAMPLE_CONFIG_PATH))
    return out


def load_config(user_path: Optional[Path] = None, example_path: Optional[Path] = None) -> Dict[str, Any]:
    """Built-in defaults merged with the FIRST config file that exists among $KB_MCP_CONFIG,
    ~/.claude/kb-mcp/config.json and config.example.json. $KB_MCP_INDEX overrides index_path."""
    cfg: Dict[str, Any] = json.loads(json.dumps(BUILTIN_CONFIG))
    loaded: Optional[Path] = None
    source = "builtin"
    for label, path in config_candidates(user_path, example_path):
        if path.is_file():
            cfg = deep_merge(cfg, _read_json(path))
            loaded, source = path, label
            break
    cfg["_paths"] = {"loaded": str(loaded) if loaded else None, "source": source,
                     "user": str(user_path or USER_CONFIG_PATH), "example": str(example_path or EXAMPLE_CONFIG_PATH)}
    env_index = os.environ.get("KB_MCP_INDEX")
    if env_index:
        cfg["index_path"] = env_index
    return cfg


def expand_user_path(raw: str) -> str:
    """~ and environment variables, forward slashes kept (SQLite stores posix-style)."""
    return os.path.expandvars(os.path.expanduser(raw))


CORPUS_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Corpus(object):
    def __init__(self, name: str, root: Optional[Path], include: List[str], exclude: List[str],
                 description: str, raw_path: str, enabled: bool = True):
        self.name = name
        self.root = root
        self.include = include
        self.exclude = exclude
        self.description = description
        self.raw_path = raw_path
        self.enabled = enabled

    @property
    def available(self) -> bool:
        return self.enabled and self.root is not None and self.root.is_dir()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root) if self.root else None,
            "configured_path": self.raw_path,
            "available": self.available,
            "enabled": self.enabled,
            "include": self.include,
            "exclude": self.exclude,
            "description": self.description,
        }


def resolve_corpora(cfg: Dict[str, Any]) -> List[Corpus]:
    """Corpora from the config: a list of {name, path, include, exclude, description, enabled}
    objects (an object keyed by name is accepted too; a null value drops the entry). `path`
    may be a folder or a single file; a file becomes a one-file corpus in its folder."""
    raw = cfg.get("corpora") or []
    specs: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(raw, dict):
        for name, spec in raw.items():
            if spec is None or spec is False:
                continue
            specs.append((str(name), dict(spec)))
    else:
        for spec in raw:
            if not isinstance(spec, dict):
                continue
            path = str(spec.get("path") or spec.get("root") or "")
            name = str(spec.get("name") or Path(expand_user_path(path)).name or "")
            specs.append((name, spec))
    default_include = list(cfg.get("default_include") or ["**/*.md"])
    out: List[Corpus] = []
    seen = set()
    for name, spec in specs:
        if not CORPUS_NAME_RE.match(name):
            log.warning("corpus %r skipped: names use letters, digits, '.', '_' and '-' only", name)
            continue
        if name in seen:
            log.warning("corpus %r defined twice; the first definition is used", name)
            continue
        seen.add(name)
        raw_path = str(spec.get("path") or spec.get("root") or "")
        root: Optional[Path] = None
        include = list(spec.get("include") or default_include)
        if raw_path:
            candidate = Path(expand_user_path(raw_path))
            if candidate.is_file():  # a single notes file is just another corpus
                root, include = candidate.parent, [candidate.name]
            else:
                root = candidate
        exclude = list(spec.get("exclude") or [])
        out.append(Corpus(name, root, include, exclude, str(spec.get("description", "")), raw_path,
                          bool(spec.get("enabled", True))))
    return out


# ----------------------------------------------------------------------------
# File discovery
# ----------------------------------------------------------------------------

def _glob_match(rel_posix: str, abs_posix: str, pattern: str) -> bool:
    pats = [pattern]
    if pattern.startswith("**/"):
        pats.append(pattern[3:])
    for pat in pats:
        if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(abs_posix, pat):
            return True
    return False


def iter_corpus_files(corpus: Corpus, global_excludes: Sequence[str], max_bytes: int) -> List[Tuple[Path, str]]:
    """(absolute path, corpus-relative posix path) for every file the corpus covers."""
    if not corpus.available or corpus.root is None:
        return []
    root = corpus.root
    seen: Dict[str, Path] = {}
    for pattern in corpus.include:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            abs_posix = path.as_posix()
            excluded = False
            for pat in list(corpus.exclude) + list(global_excludes):
                if _glob_match(rel, abs_posix, pat):
                    excluded = True
                    break
            if excluded:
                continue
            try:
                if path.stat().st_size > max_bytes:
                    log.debug("skip (too large): %s", path)
                    continue
            except OSError:
                continue
            seen[rel] = path
    return [(seen[rel], rel) for rel in sorted(seen)]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Markdown chunking
# ----------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")


class Section(object):
    __slots__ = ("level", "heading", "path", "line_start", "line_end", "lines")

    def __init__(self, level: int, heading: str, path: List[str], line_start: int):
        self.level = level
        self.heading = heading
        self.path = path
        self.line_start = line_start
        self.line_end = line_start
        self.lines: List[Tuple[int, str]] = []

    @property
    def heading_path(self) -> str:
        return " > ".join(self.path)

    @property
    def words(self) -> int:
        return sum(len(text.split()) for _, text in self.lines)

    @property
    def text(self) -> str:
        return "\n".join(text for _, text in self.lines).strip()


def parse_frontmatter(lines: List[str]) -> Tuple[Dict[str, str], int]:
    """Returns (fields, index of the first body line). Only simple `key: value` pairs."""
    if not lines or lines[0].strip() != "---":
        return {}, 0
    fields: Dict[str, str] = {}
    for i in range(1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "---":
            return fields, i + 1
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", lines[i])
        if m and not lines[i].startswith((" ", "\t")):
            value = m.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            fields[m.group(1)] = value
    return {}, 0  # unterminated: treat as body


def split_sections(text: str) -> Tuple[Dict[str, str], List[Section]]:
    """Heading-scoped sections with exact 1-based line ranges; fenced code is never a heading."""
    lines = text.splitlines()
    meta, body_start = parse_frontmatter(lines)
    sections: List[Section] = []
    stack: List[Tuple[int, str]] = []
    in_fence = False
    current = Section(0, "", [], body_start + 1)
    for idx in range(body_start, len(lines)):
        line = lines[idx]
        lineno = idx + 1
        if FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            current.line_end = lineno - 1
            sections.append(current)
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = Section(level, title, [t for _, t in stack], lineno)
            current.lines.append((lineno, line))  # keep the heading line in the body
        else:
            current.lines.append((lineno, line))
    current.line_end = len(lines)
    sections.append(current)
    # drop an empty preamble
    sections = [s for s in sections if s.text or s.level > 0]
    return meta, sections


class Chunk(object):
    __slots__ = ("heading_path", "line_start", "line_end", "body", "words", "hash")

    def __init__(self, heading_path: str, line_start: int, line_end: int, body: str):
        self.heading_path = heading_path
        self.line_start = line_start
        self.line_end = line_end
        self.body = body
        self.words = len(body.split())
        self.hash = hashlib.sha256((heading_path + "\n" + body).encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {"heading": self.heading_path, "line_start": self.line_start, "line_end": self.line_end,
                "words": self.words, "body": self.body, "hash": self.hash}


def _split_long(lines: List[Tuple[int, str]], max_words: int) -> List[List[Tuple[int, str]]]:
    """Pack paragraphs (blank-line separated) into pieces of <= max_words; a single
    oversized paragraph is cut on line boundaries."""
    paragraphs: List[List[Tuple[int, str]]] = []
    buf: List[Tuple[int, str]] = []
    for item in lines:
        if item[1].strip():
            buf.append(item)
        elif buf:
            paragraphs.append(buf)
            buf = []
    if buf:
        paragraphs.append(buf)
    pieces: List[List[Tuple[int, str]]] = []
    cur: List[Tuple[int, str]] = []
    cur_words = 0
    for para in paragraphs:
        pw = sum(len(t.split()) for _, t in para)
        if pw > max_words:
            if cur:
                pieces.append(cur)
                cur, cur_words = [], 0
            sub: List[Tuple[int, str]] = []
            sw = 0
            for item in para:
                w = len(item[1].split())
                if sub and sw + w > max_words:
                    pieces.append(sub)
                    sub, sw = [], 0
                sub.append(item)
                sw += w
            if sub:
                pieces.append(sub)
            continue
        if cur and cur_words + pw > max_words:
            pieces.append(cur)
            cur, cur_words = [], 0
        cur.extend(para)
        cur_words += pw
    if cur:
        pieces.append(cur)
    return pieces


def chunk_markdown(text: str, min_words: int = 60, target_words: int = 500, max_words: int = 800) -> Tuple[Dict[str, str], List[Chunk]]:
    """Sections become chunks; small neighbours under the same parent merge up to
    target_words; sections above max_words split on paragraph boundaries. Plain text
    without headings is one preamble section, split the same way."""
    meta, sections = split_sections(text)
    groups: List[List[Section]] = []
    for sec in sections:
        if not groups:
            groups.append([sec])
            continue
        group = groups[-1]
        first = group[0]
        group_words = sum(s.words for s in group)
        if first.path:
            descendant = sec.path[:len(first.path)] == first.path
            sibling = sec.level == first.level and sec.path[:-1] == first.path[:-1]
            related = descendant or sibling
        else:
            related = False  # a preamble never absorbs headed sections
        tiny_group = group_words < min_words  # a heading stub or a one-liner: always absorb the next section
        small = tiny_group or sec.words < min_words
        if related and (tiny_group or (small and group_words + sec.words <= target_words)):
            group.append(sec)
        else:
            groups.append([sec])
    chunks: List[Chunk] = []
    for group in groups:
        # name the chunk after its first section that has text beyond its heading line
        lead = next((s for s in group if any(t.strip() for n, t in s.lines[1:] if s.level) or (not s.level and s.text)), group[0])
        heading_path = lead.heading_path
        if not heading_path and meta.get("description"):
            heading_path = meta["description"][:160]
        lines: List[Tuple[int, str]] = []
        for sec in group:
            lines.extend(sec.lines)
        total_words = sum(len(t.split()) for _, t in lines)
        pieces = [lines] if total_words <= max_words else _split_long(lines, max_words)
        for piece in pieces:
            body = "\n".join(t for _, t in piece).strip()
            if not body:
                continue
            # trim blank edges from the line range
            nums = [n for n, t in piece if t.strip()]
            chunks.append(Chunk(heading_path, nums[0], nums[-1], body))
    return meta, chunks


def file_title(meta: Dict[str, str], sections: List[Section], rel: str) -> str:
    for sec in sections:
        if sec.level == 1:
            return sec.heading
    if meta.get("name"):
        return meta["name"]
    if meta.get("title"):
        return meta["title"]
    return Path(rel).stem


# ----------------------------------------------------------------------------
# Tokenizer (query building + pure-Python BM25 fallback)
# ----------------------------------------------------------------------------

STOPWORDS = set("""
a an and are as at be by for from has have how i in is it its of on or that the this to was we what when
where which who why will with you your do does did not no can could should would about into over under
than then there these those they them their our us me my it's don't
""".split())

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def light_stem(tok: str) -> str:
    if len(tok) > 5 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, stem: bool = True) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    out = []
    for t in toks:
        if len(t) < 2 or t in STOPWORDS:
            continue
        out.append(light_stem(t) if stem else t)
    return out


def query_terms(query: str) -> List[str]:
    """Raw (unstemmed) lowercase tokens minus stopwords; falls back to all tokens."""
    raw = [t.lower() for t in TOKEN_RE.findall(query)]
    kept = [t for t in raw if len(t) >= 2 and t not in STOPWORDS]
    return kept or raw


# ----------------------------------------------------------------------------
# Embeddings (Ollama /api/embed)
# ----------------------------------------------------------------------------

class EmbedError(Exception):
    pass


def pace_request(url: str) -> None:
    """Hook called before every HTTP request to the embedder. A no-op here: the default
    endpoint is the local machine. If you point `embedder.url` at a shared or public host and
    want to space or count requests, put that logic here (block until it is this request's
    turn, or raise EmbedError to refuse)."""
    return None


def report_response(url: str, status: int) -> None:
    """Hook called with the HTTP status of a failed embedder request (a back-off could live here)."""
    return None


def pack_vector(vec: Sequence[float]) -> bytes:
    arr = array("f", vec)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def unpack_vector(blob: bytes) -> array:
    arr = array("f")
    arr.frombytes(blob)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr


def normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def default_prefixes(model: str) -> Tuple[str, str]:
    """nomic-embed-text wants task prefixes; other models usually want none."""
    if model.split(":")[0].startswith("nomic-embed"):
        return "search_document: ", "search_query: "
    return "", ""


class Embedder(object):
    """Ollama /api/embed client. available() is cheap and cached; embed() raises EmbedError."""

    def __init__(self, cfg: Dict[str, Any]):
        emb = cfg.get("embedder", {})
        self.enabled = bool(emb.get("enabled", True))
        self.url = str(emb.get("url", "http://127.0.0.1:11434")).rstrip("/")
        self.model = str(emb.get("model", "nomic-embed-text"))
        doc_default, query_default = default_prefixes(self.model)
        self.doc_prefix = str(emb["doc_prefix"]) if emb.get("doc_prefix") is not None else doc_default
        self.query_prefix = str(emb["query_prefix"]) if emb.get("query_prefix") is not None else query_default
        self.batch_size = int(emb.get("batch_size", 32))
        self.timeout = float(emb.get("timeout_s", 120))
        self.probe_timeout = float(emb.get("probe_timeout_s", 2))
        self.max_chars = int(emb.get("max_chars", 6000))
        self.num_ctx = int(emb.get("num_ctx", 8192))
        self._probe: Tuple[float, bool, str] = (0.0, False, "not probed")
        self.probe_ttl = 30.0
        self.failed_texts = 0  # inputs no cap could make the model accept (left pending)

    @property
    def signature(self) -> str:
        return "%s|%s" % (self.model, self.doc_prefix)

    def available(self, force: bool = False) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "embedder disabled in config (lexical search only)"
        now = time.time()
        if not force and now - self._probe[0] < self.probe_ttl:
            return self._probe[1], self._probe[2]
        ok, reason = False, ""
        try:
            pace_request(self.url + "/api/tags")
            with urlrequest.urlopen(self.url + "/api/tags", timeout=self.probe_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in data.get("models", [])]
            base = self.model.split(":")[0]
            if any(n == self.model or n.split(":")[0] == base for n in names):
                ok, reason = True, "embedder up, model present"
            else:
                reason = "embedder up at %s but model %s is not pulled (ollama pull %s)" % (self.url, self.model, self.model)
        except Exception as exc:  # URLError, timeout, JSON errors
            reason = "embedder unreachable at %s (%s)" % (self.url, exc.__class__.__name__)
        self._probe = (now, ok, reason)
        return ok, reason

    def _post(self, texts: List[str]) -> List[List[float]]:
        url = self.url + "/api/embed"
        body: Dict[str, Any] = {"model": self.model, "input": texts, "truncate": True}
        if self.num_ctx:
            body["options"] = {"num_ctx": self.num_ctx}
        payload = json.dumps(body).encode("utf-8")
        req = urlrequest.Request(url, data=payload, headers={"Content-Type": "application/json"})
        pace_request(url)
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            report_response(url, exc.code)
            self._probe = (time.time(), False, "embed call failed (HTTP %s)" % exc.code)
            raise EmbedError("HTTP %s: %s" % (exc.code, detail or exc.reason))
        except Exception as exc:
            self._probe = (time.time(), False, "embed call failed (%s)" % exc.__class__.__name__)
            raise EmbedError(str(exc))
        vectors = data.get("embeddings")
        if vectors is None and data.get("embedding") is not None:
            vectors = [data["embedding"]]
        if not vectors or len(vectors) != len(texts):
            raise EmbedError("unexpected response shape from /api/embed")
        return [normalize(v) for v in vectors]

    @staticmethod
    def _is_length_error(exc: EmbedError) -> bool:
        msg = str(exc).lower()
        return "context length" in msg or "input length" in msg or msg.startswith("http 400")

    def embed_documents(self, texts: List[str]) -> List[Optional[List[float]]]:
        """One vector per text, or None for an input the model rejects at every cap.
        Connection failures still raise EmbedError (the caller stops the run)."""
        out: List[Optional[List[float]]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            batch = [self.doc_prefix + t[: self.max_chars] for t in chunk]
            try:
                out.extend(self._post(batch))
                continue
            except EmbedError as exc:
                if not self._is_length_error(exc):
                    raise
            # one input in the batch is too long for the model: retry singly with a shrinking cap
            for text in chunk:
                vec: Optional[List[float]] = None
                for cap in (self.max_chars, self.max_chars // 2, self.max_chars // 4):
                    try:
                        vec = self._post([self.doc_prefix + text[:cap]])[0]
                        break
                    except EmbedError as exc:
                        if not self._is_length_error(exc):
                            raise
                if vec is None:
                    self.failed_texts += 1
                    log.warning("embedding rejected at every cap (%d chars); left pending", len(text))
                out.append(vec)
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._post([self.query_prefix + text[: self.max_chars]])[0]


# ----------------------------------------------------------------------------
# Store (SQLite)
# ----------------------------------------------------------------------------

def fts5_available() -> bool:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        con.close()
        return True
    except sqlite3.Error:
        return False


class Store(object):
    def __init__(self, db_path: Path, lexical_backend: str = "auto"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA busy_timeout=5000")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.has_fts5 = fts5_available()
        if lexical_backend == "auto":
            lexical_backend = "fts5" if self.has_fts5 else "python"
        if lexical_backend == "fts5" and not self.has_fts5:
            log.warning("FTS5 requested but unavailable in this sqlite3; using the Python BM25 backend")
            lexical_backend = "python"
        self.backend = lexical_backend
        self._vec_cache: Optional[Tuple[int, str, List[Tuple[int, str, str, array]]]] = None
        self.ensure_schema()

    # -- schema -----------------------------------------------------------
    def ensure_schema(self) -> None:
        c = self.con
        c.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                corpus TEXT NOT NULL,
                rel TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                title TEXT,
                indexed_at REAL NOT NULL,
                UNIQUE(corpus, rel)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                corpus TEXT NOT NULL,
                rel TEXT NOT NULL,
                heading TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                words INTEGER NOT NULL,
                body TEXT NOT NULL,
                hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                sig TEXT NOT NULL,
                vec BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS postings (
                term TEXT NOT NULL,
                chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                tf INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
            CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);
        """)
        c.execute("PRAGMA foreign_keys=ON")
        if self.backend == "fts5":
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                      "heading, body, path, tokenize='porter unicode61')")
        stored = self.get_meta("schema_version")
        if stored is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        stored_backend = self.get_meta("lexical_backend")
        if stored_backend is None:
            self.set_meta("lexical_backend", self.backend)
        elif stored_backend != self.backend:
            log.warning("lexical backend changed (%s -> %s); rebuilding the lexical index", stored_backend, self.backend)
            self.rebuild_lexical()
            self.set_meta("lexical_backend", self.backend)
        c.commit()

    def close(self) -> None:
        try:
            self.con.close()
        except sqlite3.Error:
            pass

    # -- meta -------------------------------------------------------------
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def bump_embed_gen(self) -> None:
        gen = int(self.get_meta("embed_gen", "0") or 0) + 1
        self.set_meta("embed_gen", str(gen))

    # -- files / chunks ---------------------------------------------------
    def file_record(self, corpus: str, rel: str) -> Optional[sqlite3.Row]:
        return self.con.execute("SELECT * FROM files WHERE corpus=? AND rel=?", (corpus, rel)).fetchone()

    def file_records(self, corpus: str) -> Dict[str, sqlite3.Row]:
        rows = self.con.execute("SELECT * FROM files WHERE corpus=?", (corpus,)).fetchall()
        return {r["rel"]: r for r in rows}

    def _lexical_insert(self, chunk_id: int, heading: str, body: str, path: str) -> None:
        if self.backend == "fts5":
            self.con.execute("INSERT INTO chunks_fts(rowid, heading, body, path) VALUES (?, ?, ?, ?)",
                             (chunk_id, heading, body, path))
        else:
            counts: Dict[str, int] = defaultdict(int)
            for t in tokenize(body):
                counts[t] += 1
            for t in tokenize(heading):
                counts[t] += 2
            for t in tokenize(path.replace("/", " ").replace(".", " ")):
                counts[t] += 1
            self.con.executemany("INSERT INTO postings(term, chunk_id, tf) VALUES (?, ?, ?)",
                                 [(t, chunk_id, n) for t, n in counts.items()])

    def _lexical_delete(self, chunk_ids: List[int]) -> None:
        if not chunk_ids:
            return
        if self.backend == "fts5":
            self.con.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in chunk_ids])
        else:
            self.con.executemany("DELETE FROM postings WHERE chunk_id=?", [(i,) for i in chunk_ids])

    def rebuild_lexical(self) -> None:
        if self.backend == "fts5":
            self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                             "heading, body, path, tokenize='porter unicode61')")
            self.con.execute("DELETE FROM chunks_fts")
        else:
            self.con.execute("DELETE FROM postings")
        rows = self.con.execute("SELECT id, heading, body, corpus, rel FROM chunks").fetchall()
        for r in rows:
            self._lexical_insert(r["id"], r["heading"], r["body"], r["corpus"] + "/" + r["rel"])

    def replace_file(self, corpus: str, rel: str, path: Path, sha: str, mtime: float, size: int,
                     title: str, chunks: List[Chunk], sig: str) -> Tuple[int, int]:
        """Replace a file's chunks; embeddings of unchanged chunks (same hash) carry over.
        Returns (chunks written, embeddings carried over)."""
        old = self.file_record(corpus, rel)
        carried: Dict[str, bytes] = {}
        if old is not None:
            rows = self.con.execute(
                "SELECT c.id, c.hash, e.vec FROM chunks c LEFT JOIN embeddings e "
                "ON e.chunk_id=c.id AND e.sig=? WHERE c.file_id=?", (sig, old["id"])).fetchall()
            for r in rows:
                if r["vec"] is not None:
                    carried[r["hash"]] = r["vec"]
            self._lexical_delete([r["id"] for r in rows])
            self.con.execute("DELETE FROM chunks WHERE file_id=?", (old["id"],))
            self.con.execute("UPDATE files SET path=?, sha256=?, mtime=?, size=?, title=?, indexed_at=? WHERE id=?",
                             (path.as_posix(), sha, mtime, size, title, time.time(), old["id"]))
            file_id = old["id"]
        else:
            cur = self.con.execute(
                "INSERT INTO files(corpus, rel, path, sha256, mtime, size, title, indexed_at) VALUES (?,?,?,?,?,?,?,?)",
                (corpus, rel, path.as_posix(), sha, mtime, size, title, time.time()))
            file_id = cur.lastrowid
        reused = 0
        for ch in chunks:
            cur = self.con.execute(
                "INSERT INTO chunks(file_id, corpus, rel, heading, line_start, line_end, words, body, hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (file_id, corpus, rel, ch.heading_path, ch.line_start, ch.line_end, ch.words, ch.body, ch.hash))
            cid = cur.lastrowid
            self._lexical_insert(cid, ch.heading_path, ch.body, corpus + "/" + rel)
            if ch.hash in carried:
                self.con.execute("INSERT INTO embeddings(chunk_id, sig, vec) VALUES (?,?,?)",
                                 (cid, sig, carried[ch.hash]))
                reused += 1
        if old is not None or reused:
            self.bump_embed_gen()  # the vector set changed (old chunks gone, carried ones re-keyed)
        return len(chunks), reused

    def delete_file(self, corpus: str, rel: str) -> int:
        old = self.file_record(corpus, rel)
        if old is None:
            return 0
        ids = [r[0] for r in self.con.execute("SELECT id FROM chunks WHERE file_id=?", (old["id"],))]
        self._lexical_delete(ids)
        self.con.execute("DELETE FROM files WHERE id=?", (old["id"],))
        if ids:
            self.bump_embed_gen()
        return len(ids)

    def delete_corpus(self, corpus: str) -> int:
        ids = [r[0] for r in self.con.execute("SELECT id FROM chunks WHERE corpus=?", (corpus,))]
        self._lexical_delete(ids)
        self.con.execute("DELETE FROM files WHERE corpus=?", (corpus,))
        if ids:
            self.bump_embed_gen()
        return len(ids)

    # -- embeddings -------------------------------------------------------
    def chunks_missing_embeddings(self, sig: str, limit: Optional[int] = None) -> List[sqlite3.Row]:
        sql = ("SELECT c.id, c.corpus, c.rel, c.heading, c.body FROM chunks c "
               "LEFT JOIN embeddings e ON e.chunk_id=c.id AND e.sig=? WHERE e.chunk_id IS NULL ORDER BY c.id")
        if limit:
            sql += " LIMIT %d" % int(limit)
        return self.con.execute(sql, (sig,)).fetchall()

    def put_embeddings(self, items: Iterable[Tuple[int, Optional[Sequence[float]]]], sig: str) -> int:
        n = 0
        for cid, vec in items:
            if vec is None:
                continue
            self.con.execute("INSERT OR REPLACE INTO embeddings(chunk_id, sig, vec) VALUES (?,?,?)",
                             (cid, sig, pack_vector(vec)))
            n += 1
        if n:
            self.bump_embed_gen()
        return n

    def drop_foreign_embeddings(self, sig: str) -> int:
        cur = self.con.execute("DELETE FROM embeddings WHERE sig<>?", (sig,))
        if cur.rowcount:
            self.bump_embed_gen()
        return cur.rowcount

    def embedding_coverage(self, sig: str) -> Tuple[int, int]:
        total = self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        done = self.con.execute("SELECT COUNT(*) FROM embeddings WHERE sig=?", (sig,)).fetchone()[0]
        return done, total

    def _vectors(self, sig: str) -> List[Tuple[int, str, str, array]]:
        gen = int(self.get_meta("embed_gen", "0") or 0)
        if self._vec_cache and self._vec_cache[0] == gen and self._vec_cache[1] == sig:
            return self._vec_cache[2]
        rows = self.con.execute(
            "SELECT e.chunk_id, c.corpus, c.rel, e.vec FROM embeddings e JOIN chunks c ON c.id=e.chunk_id WHERE e.sig=?",
            (sig,)).fetchall()
        vectors = [(r[0], r[1], r[2], unpack_vector(r[3])) for r in rows]
        self._vec_cache = (gen, sig, vectors)
        return vectors

    def vector_search(self, qvec: Sequence[float], sig: str, pool: int,
                      corpora: Optional[Sequence[str]] = None) -> List[Tuple[int, float]]:
        q = array("f", qvec)
        scored: List[Tuple[int, float]] = []
        allowed = set(corpora) if corpora else None
        for cid, corpus, _rel, vec in self._vectors(sig):
            if allowed is not None and corpus not in allowed:
                continue
            if len(vec) != len(q):
                continue
            scored.append((cid, sum(map(mul, q, vec))))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:pool]

    # -- lexical ----------------------------------------------------------
    def lexical_search(self, query: str, pool: int, corpora: Optional[Sequence[str]] = None) -> List[Tuple[int, float]]:
        if self.backend == "fts5":
            return self._fts_search(query, pool, corpora)
        return self._python_bm25(query, pool, corpora)

    def _fts_search(self, query: str, pool: int, corpora: Optional[Sequence[str]]) -> List[Tuple[int, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        parts = ['"%s"' % t.replace('"', '') for t in terms]
        raw_tokens = [t.lower() for t in TOKEN_RE.findall(query)]
        if len(raw_tokens) >= 2:
            parts.insert(0, '"%s"' % " ".join(raw_tokens).replace('"', ''))
        match = " OR ".join(parts)
        sql = ("SELECT f.rowid AS id, bm25(chunks_fts, 2.0, 1.0, 0.5) AS score FROM chunks_fts f "
               "JOIN chunks c ON c.id=f.rowid WHERE chunks_fts MATCH ?")
        params: List[Any] = [match]
        if corpora:
            sql += " AND c.corpus IN (%s)" % ",".join("?" * len(corpora))
            params.extend(corpora)
        sql += " ORDER BY score LIMIT ?"
        params.append(pool)
        try:
            rows = self.con.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            log.debug("fts query failed (%s); retrying with plain terms", exc)
            match = " OR ".join('"%s"' % t for t in terms)
            params[0] = match
            rows = self.con.execute(sql, params).fetchall()
        return [(r[0], -float(r[1])) for r in rows]  # FTS5 bm25 is negative-is-better

    def _python_bm25(self, query: str, pool: int, corpora: Optional[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> List[Tuple[int, float]]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        where = ""
        params: List[Any] = []
        if corpora:
            where = " WHERE corpus IN (%s)" % ",".join("?" * len(corpora))
            params = list(corpora)
        row = self.con.execute("SELECT COUNT(*), AVG(words) FROM chunks" + where, params).fetchone()
        n_docs = row[0] or 0
        avgdl = float(row[1] or 1.0)
        if not n_docs:
            return []
        allowed = set(corpora) if corpora else None
        scores: Dict[int, float] = defaultdict(float)
        for term in terms:
            rows = self.con.execute(
                "SELECT p.chunk_id, p.tf, c.words, c.corpus FROM postings p JOIN chunks c ON c.id=p.chunk_id WHERE p.term=?",
                (term,)).fetchall()
            rows = [r for r in rows if allowed is None or r[3] in allowed]
            df = len(rows)
            if not df:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for cid, tf, words, _corpus in rows:
                denom = tf + k1 * (1.0 - b + b * (words / avgdl))
                scores[cid] += idf * (tf * (k1 + 1.0)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:pool]

    # -- reads ------------------------------------------------------------
    def chunk_rows(self, ids: Sequence[int]) -> Dict[int, sqlite3.Row]:
        if not ids:
            return {}
        rows = self.con.execute(
            "SELECT c.*, f.path AS abs_path, f.mtime AS mtime, f.title AS title FROM chunks c JOIN files f ON f.id=c.file_id "
            "WHERE c.id IN (%s)" % ",".join("?" * len(ids)), list(ids)).fetchall()
        return {r["id"]: r for r in rows}

    def recent(self, corpus: Optional[str], days: float, limit: int) -> List[sqlite3.Row]:
        since = time.time() - days * 86400.0
        sql = ("SELECT f.*, (SELECT COUNT(*) FROM chunks c WHERE c.file_id=f.id) AS chunks "
               "FROM files f WHERE f.mtime >= ?")
        params: List[Any] = [since]
        if corpus:
            sql += " AND f.corpus=?"
            params.append(corpus)
        sql += " ORDER BY f.mtime DESC LIMIT ?"
        params.append(limit)
        return self.con.execute(sql, params).fetchall()

    def corpus_stats(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in self.con.execute("SELECT corpus, COUNT(*) AS files FROM files GROUP BY corpus"):
            out.setdefault(r["corpus"], {})["files"] = r["files"]
        for r in self.con.execute("SELECT corpus, COUNT(*) AS chunks, SUM(words) AS words FROM chunks GROUP BY corpus"):
            out.setdefault(r["corpus"], {})["chunks"] = r["chunks"]
            out.setdefault(r["corpus"], {})["words"] = r["words"] or 0
        for r in self.con.execute(
                "SELECT c.corpus, COUNT(*) AS n FROM embeddings e JOIN chunks c ON c.id=e.chunk_id GROUP BY c.corpus"):
            out.setdefault(r["corpus"], {})["embedded"] = r["n"]
        return out

    def find_file(self, spec: str) -> List[sqlite3.Row]:
        """Resolve 'corpus:rel', an absolute path, or a bare rel path against indexed files."""
        spec = spec.strip()
        if ":" in spec and not re.match(r"^[A-Za-z]:[\\/]", spec):
            corpus, rel = spec.split(":", 1)
            row = self.file_record(corpus.strip(), rel.strip().lstrip("/"))
            return [row] if row else []
        posix = Path(expand_user_path(spec)).as_posix() if spec else spec
        rows = self.con.execute("SELECT * FROM files WHERE path=?", (posix,)).fetchall()
        if rows:
            return rows
        rows = self.con.execute("SELECT * FROM files WHERE rel=?", (spec.lstrip("/"),)).fetchall()
        if rows:
            return rows
        # suffix match: 'tools/setup.md' or 'setup.md'
        rows = self.con.execute("SELECT * FROM files WHERE rel LIKE ?", ("%" + spec.lstrip("/"),)).fetchall()
        return rows


# ----------------------------------------------------------------------------
# Index lock
# ----------------------------------------------------------------------------

def _pid_alive(pid: int, created: Optional[float] = None) -> Optional[bool]:
    """Is the process that wrote an index lock still running? True or False, or None when this
    platform cannot tell. `created` is the lock's creation time: on Windows, where PIDs are reused
    quickly, a process with that PID that started after it cannot be the owner. POSIX hands PIDs
    out in sequence, so a reused PID there is left to the lock's age limit."""
    if pid <= 0:
        return None  # os.kill reads 0 and negative numbers as process groups
    if os.name == "nt":
        if pid > 0xFFFFFFFF:  # beyond DWORD: a corrupt record; ctypes would mask it onto another PID
            return None
        return _pid_alive_windows(pid, created)
    try:
        os.kill(pid, 0)  # signal 0 only checks; never on Windows, where os.kill terminates the process
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except (OSError, OverflowError):  # OverflowError: a corrupt record with a PID beyond pid_t
        return None
    return True


def _pid_alive_windows(pid: int, created: Optional[float]) -> Optional[bool]:
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k32.GetProcessTimes.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            err = ctypes.get_last_error()
            if err == 87:  # ERROR_INVALID_PARAMETER: no process has this PID
                return False
            if err == 5:  # ERROR_ACCESS_DENIED: it exists
                return True
            return None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            if code.value != 259:  # not STILL_ACTIVE: exited, kept as an object by an open handle
                return False
            if created is not None:
                times = [wintypes.FILETIME() for _ in range(4)]
                if k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                    ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
                    started = ticks / 1e7 - 11644473600.0  # FILETIME (100 ns since 1601) -> Unix time
                    if started > created + 2.0:
                        return False  # the PID now belongs to a process younger than the lock
            return True
        finally:
            k32.CloseHandle(handle)
    except Exception:  # no ctypes, or an API surprise: cannot tell
        return None


class IndexLock(object):
    """One index writer per database: an O_EXCL lock file holding {"pid", "created"}. It is taken
    over when its owner is gone (_pid_alive) or the file is older than stale_s; a lock whose owner
    cannot be checked is respected until then. A server killed in the middle of a run therefore
    blocks nobody. Takeover and release compare the file's bytes first, so neither removes a lock
    that another run took in the meantime."""

    def __init__(self, db_path: Path, stale_s: float = 900.0):
        self.path = db_path.with_suffix(".lock")
        self.stale_s = stale_s
        self.acquired = False
        self._token = b""

    @staticmethod
    def parse(raw: bytes) -> Tuple[Optional[int], Optional[float]]:
        """(pid, created) from the lock's bytes: a JSON object, or a plain "pid created" text record."""
        try:
            text = raw.decode("ascii").strip()
            if text.startswith("{"):
                data = json.loads(text)
                return int(data["pid"]), float(data["created"])
            pid, created = text.split()[:2]
            return int(pid), float(created)
        except (ValueError, KeyError, TypeError):
            return None, None

    def _read(self) -> Tuple[Optional[bytes], float]:
        try:
            age = time.time() - os.stat(str(self.path)).st_mtime
            with open(str(self.path), "rb") as fh:
                return fh.read(4096), age
        except OSError:
            return None, 0.0

    def stale_reason(self, raw: bytes, age: float) -> Optional[str]:
        """Why a lock with these bytes and this age may be taken over; None while it must be respected
        (a live owner, or content not written yet / unreadable, until stale_s)."""
        if age > self.stale_s:
            return "%.0fs old" % age
        pid, created = self.parse(raw)
        if pid is not None and _pid_alive(pid, created) is False:
            return "owner pid %d is gone" % pid
        return None

    def _break(self, raw: bytes) -> bool:
        """Remove the stale lock whose bytes are `raw`. It is renamed aside first (atomic), so of several
        runs judging the same lock only one removes it, and a lock that is not the one judged (another
        run took over in between) is put back. False when the file could not be moved."""
        claim = self.path.with_name("%s.%d.stale" % (self.path.name, os.getpid()))
        try:
            os.replace(str(self.path), str(claim))
        except OSError:
            return False  # gone already, or open elsewhere for a moment (Windows)
        try:
            with open(str(claim), "rb") as fh:
                moved = fh.read(4096)  # type: Optional[bytes]
        except OSError:
            moved = None
        if moved != raw:
            try:  # both calls fail instead of overwriting a lock that exists again
                if os.name == "nt":
                    os.rename(str(claim), str(self.path))
                else:
                    os.link(str(claim), str(self.path))
            except OSError:
                pass
        try:
            os.unlink(str(claim))
        except OSError:
            pass
        return True

    def __enter__(self):
        for _ in range(3):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raw, age = self._read()
                if raw is None:
                    continue  # removed between the two calls: try again
                why = self.stale_reason(raw, age)
                if why is None:
                    return self  # held by a live run
                if why.startswith("owner"):
                    log.info("taking over the index lock: %s", why)  # routine after a killed server
                else:
                    log.warning("taking over the index lock: %s", why)
                if not self._break(raw):
                    time.sleep(0.05)
                continue
            token = json.dumps({"pid": os.getpid(), "created": time.time()}).encode("ascii")
            try:
                os.write(fd, token)
            finally:
                os.close(fd)
            self._token = token
            self.acquired = True
            return self
        return self

    def __exit__(self, *exc):
        if not self.acquired:
            return
        self.acquired = False
        raw, _age = self._read()
        if raw is None:
            return
        if raw != self._token:
            log.warning("the index lock was taken over during this run; leaving the other run's lock in place")
            return
        for _ in range(3):
            try:
                os.unlink(str(self.path))
                return
            except FileNotFoundError:
                return
            except OSError:
                time.sleep(0.05)  # open elsewhere for a moment (Windows)


# ----------------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------------

def index(cfg: Dict[str, Any], store: Store, embedder: Optional[Embedder] = None, full: bool = False,
          corpora: Optional[Sequence[str]] = None, embed: bool = True, progress=None,
          should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Incremental index. A file is re-chunked when its sha256 changed (mtime+size are the
    cheap pre-check); removed files are purged; new chunks get embeddings when the embedder answers.

    should_stop is checked before each corpus, file and embedding batch; when it returns True the
    run commits what it has done, releases the lock and returns with report["stopped"] (no purge
    from a partial walk, and the index is not marked fresh, so the next run finishes the job)."""
    t0 = time.time()
    stop = should_stop or (lambda: False)
    report: Dict[str, Any] = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "full": full, "corpora": {}, "files_seen": 0, "files_indexed": 0, "files_unchanged": 0,
        "files_removed": 0, "chunks_written": 0, "embeddings_reused": 0, "embeddings_new": 0,
        "embeddings_pending": 0, "embedding_mode": "skipped", "embedding_reason": "", "skipped": False,
        "stopped": False,
    }
    all_corpora = resolve_corpora(cfg)
    wanted = [c for c in all_corpora if not corpora or c.name in corpora]
    chunk_cfg = cfg.get("chunking", {})
    excludes = cfg.get("exclude_globs", [])
    max_bytes = int(cfg.get("max_file_bytes", 2000000))
    sig = embedder.signature if embedder else (store.get_meta("embed_sig") or "")

    with IndexLock(store.db_path) as lock:
        if not lock.acquired:
            report["skipped"] = True
            report["reason"] = "another index run holds the lock"
            report["seconds"] = round(time.time() - t0, 3)
            return report
        for corpus in wanted:
            if report["stopped"] or stop():
                report["stopped"] = True
                break
            cstat = {"root": str(corpus.root) if corpus.root else None, "available": corpus.available,
                     "files": 0, "indexed": 0, "unchanged": 0, "removed": 0, "chunks": 0}
            report["corpora"][corpus.name] = cstat
            if not corpus.available:
                cstat["note"] = "disabled" if not corpus.enabled else "folder not found"
                continue
            existing = store.file_records(corpus.name)
            present = set()
            files = iter_corpus_files(corpus, excludes, max_bytes)
            cstat["files"] = len(files)
            report["files_seen"] += len(files)
            for path, rel in files:
                if stop():
                    report["stopped"] = True
                    break
                present.add(rel)
                try:
                    st = path.stat()
                except OSError:
                    continue
                old = existing.get(rel)
                if old is not None and not full and abs(old["mtime"] - st.st_mtime) < 1e-6 and old["size"] == st.st_size:
                    cstat["unchanged"] += 1
                    report["files_unchanged"] += 1
                    continue
                sha = file_sha256(path)
                if old is not None and not full and old["sha256"] == sha:
                    store.con.execute("UPDATE files SET mtime=?, size=? WHERE id=?", (st.st_mtime, st.st_size, old["id"]))
                    cstat["unchanged"] += 1
                    report["files_unchanged"] += 1
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    log.warning("cannot read %s: %s", path, exc)
                    continue
                meta, chunks = chunk_markdown(text, int(chunk_cfg.get("min_words", 60)),
                                              int(chunk_cfg.get("target_words", 500)), int(chunk_cfg.get("max_words", 800)))
                _, sections = split_sections(text)
                title = file_title(meta, sections, rel)
                n, reused = store.replace_file(corpus.name, rel, path, sha, st.st_mtime, st.st_size, title, chunks, sig)
                cstat["indexed"] += 1
                cstat["chunks"] += n
                report["files_indexed"] += 1
                report["chunks_written"] += n
                report["embeddings_reused"] += reused
                if progress:
                    progress("indexed %s:%s (%d chunks)" % (corpus.name, rel, n))
            for rel in (set() if report["stopped"] else set(existing) - present):  # a partial walk purges nothing
                store.delete_file(corpus.name, rel)
                cstat["removed"] += 1
                report["files_removed"] += 1
            store.con.commit()

        # corpora that were removed from the config are purged (nothing in the config may be left half-indexed)
        if not corpora and not report["stopped"]:
            configured = {c.name for c in all_corpora}
            for name in list(store.corpus_stats()):
                if name not in configured:
                    removed = store.delete_corpus(name)
                    report["corpora"][name] = {"note": "no longer configured; purged", "removed": removed}
                    log.info("corpus %s is no longer configured; %d chunks purged", name, removed)
            store.con.commit()

        # embeddings for whatever lacks one (this run's new chunks + any backlog)
        if embedder is not None and embed and embedder.enabled and not report["stopped"]:
            ok, reason = embedder.available(force=True)
            report["embedding_reason"] = reason
            if ok:
                if store.get_meta("embed_sig") not in (None, sig):
                    dropped = store.drop_foreign_embeddings(sig)
                    log.info("embedding signature changed; dropped %d old vectors", dropped)
                store.set_meta("embed_sig", sig)
                pending = store.chunks_missing_embeddings(sig)
                te = time.time()
                done = 0
                rejected_before = embedder.failed_texts
                try:
                    for i in range(0, len(pending), embedder.batch_size):
                        if stop():
                            report["stopped"] = True
                            break
                        batch = pending[i:i + embedder.batch_size]
                        texts = ["%s/%s\n%s\n\n%s" % (r["corpus"], r["rel"], r["heading"], r["body"]) for r in batch]
                        vectors = embedder.embed_documents(texts)
                        done += store.put_embeddings(zip([r["id"] for r in batch], vectors), sig)
                        store.con.commit()
                        if progress:
                            progress("embedded %d/%d" % (done, len(pending)))
                    report["embedding_mode"] = "stopped" if report["stopped"] else "ollama"
                except EmbedError as exc:
                    report["embedding_mode"] = "interrupted"
                    report["embedding_reason"] = "embed call failed: %s" % exc
                report["embeddings_new"] = done
                report["embeddings_rejected"] = embedder.failed_texts - rejected_before
                report["embedding_seconds"] = round(time.time() - te, 3)
            else:
                report["embedding_mode"] = "unavailable"
            report["embeddings_pending"] = len(store.chunks_missing_embeddings(sig))
        elif embedder is not None:
            report["embedding_mode"] = "disabled" if not embedder.enabled else "skipped"
            report["embeddings_pending"] = len(store.chunks_missing_embeddings(sig))

        if report["stopped"]:
            report["reason"] = "stopped before the end; the next run indexes the rest"
            store.con.commit()
            report["seconds"] = round(time.time() - t0, 3)
            return report
        store.set_meta("last_index_run", str(time.time()))
        store.set_meta("last_index_report", json.dumps({k: v for k, v in report.items() if k != "corpora"}))
        store.con.commit()
        if full:
            try:
                store.con.execute("VACUUM")  # a full run churns every row; reclaim the space
            except sqlite3.Error as exc:
                log.debug("vacuum skipped: %s", exc)
    report["seconds"] = round(time.time() - t0, 3)
    return report


def index_age_seconds(store: Store) -> Optional[float]:
    raw = store.get_meta("last_index_run")
    if not raw:
        return None
    try:
        return time.time() - float(raw)
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------

def rrf_fuse(lanes: Sequence[Sequence[Tuple[int, float]]], k: int = 20,
             weights: Optional[Sequence[float]] = None) -> List[Tuple[int, float, List[Optional[int]]]]:
    """Weighted reciprocal rank fusion: score = sum_i w_i / (k + rank_i).
    Returns (id, score, [rank in lane i or None]) best first."""
    weights = list(weights) if weights else [1.0] * len(lanes)
    scores: Dict[int, float] = defaultdict(float)
    ranks: Dict[int, List[Optional[int]]] = defaultdict(lambda: [None] * len(lanes))
    for li, lane in enumerate(lanes):
        for rank, (cid, _score) in enumerate(lane, start=1):
            scores[cid] += weights[li] / (k + rank)
            ranks[cid][li] = rank
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(cid, score, ranks[cid]) for cid, score in ordered]


def lexical_coverage(query: str, row: Optional[sqlite3.Row]) -> float:
    """Share of the query's content terms present in the lexical top hit (heading, body, path).
    1.0 = the lexical lane matched every term, so it is trusted; low = paraphrase, lean on vectors."""
    if row is None:
        return 0.0
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    doc = set(tokenize("%s %s %s" % (row["heading"], row["body"], row["rel"].replace("/", " ").replace("_", " "))))
    return len(terms & doc) / float(len(terms))


def make_snippet(body: str, terms: Sequence[str], max_chars: int) -> str:
    flat = re.sub(r"\s+", " ", body).strip()
    if len(flat) <= max_chars:
        return flat
    low = flat.lower()
    best = -1
    for t in terms:
        probe = t[: max(4, len(t) - 2)] if len(t) > 4 else t
        pos = low.find(probe)
        if pos != -1 and (best == -1 or pos < best):
            best = pos
    if best == -1:
        return flat[:max_chars].rstrip() + "..."
    start = max(0, best - max_chars // 4)
    if start:
        space = flat.rfind(" ", 0, start)
        start = space + 1 if space != -1 else start
    end = min(len(flat), start + max_chars)
    piece = flat[start:end].strip()
    return ("..." if start else "") + piece + ("..." if end < len(flat) else "")


def search(query: str, cfg: Dict[str, Any], store: Store, embedder: Optional[Embedder], k: Optional[int] = None,
           corpora: Optional[Sequence[str]] = None, mode: str = "auto") -> Dict[str, Any]:
    scfg = cfg.get("search", {})
    k = int(k or scfg.get("k", 6))
    pool = max(int(scfg.get("pool", 40)), k * 3)
    vec_pool = max(int(scfg.get("vec_pool", 20)), k)
    rrf_k = int(scfg.get("rrf_k", 20))
    w_lo = float(scfg.get("vec_weight_min", 0.2))
    w_hi = float(scfg.get("vec_weight_max", 1.0))
    snippet_chars = int(scfg.get("snippet_chars", 600))
    max_per_file = int(scfg.get("max_per_file", 3))
    timings: Dict[str, float] = {}
    note = ""

    t = time.time()
    lex = store.lexical_search(query, pool, corpora)
    timings["lexical_ms"] = round((time.time() - t) * 1000, 1)
    top_row = store.chunk_rows([lex[0][0]]).get(lex[0][0]) if lex else None
    coverage = lexical_coverage(query, top_row)
    vec_weight = w_hi - (w_hi - w_lo) * coverage

    vec: List[Tuple[int, float]] = []
    used_mode = "lexical"
    lexical_label = "fts5" if store.backend == "fts5" else "python-bm25"
    if mode != "lexical" and embedder is not None and embedder.enabled:
        sig = embedder.signature
        done, total = store.embedding_coverage(sig)
        if done == 0:
            note = "no embeddings in the index yet"
        else:
            ok, reason = embedder.available()
            if not ok:
                note = reason
            else:
                try:
                    t = time.time()
                    qvec = embedder.embed_query(query)
                    timings["embed_ms"] = round((time.time() - t) * 1000, 1)
                    t = time.time()
                    vec = store.vector_search(qvec, sig, vec_pool, corpora)
                    timings["vector_ms"] = round((time.time() - t) * 1000, 1)
                    used_mode = "hybrid"
                    if done < total:
                        note = "embeddings cover %d/%d chunks" % (done, total)
                except EmbedError as exc:
                    note = "query embedding failed (%s)" % exc
    if mode == "hybrid" and used_mode != "hybrid":
        note = (note + "; " if note else "") + "hybrid requested but answered lexically"

    fused = rrf_fuse([lex, vec], rrf_k, [1.0, vec_weight])  # an empty vector lane contributes nothing
    lex_scores = dict(lex)
    vec_scores = dict(vec)
    rows = store.chunk_rows([cid for cid, _s, _r in fused[: pool * 2]])
    terms = query_terms(query)
    hits: List[Dict[str, Any]] = []
    per_file: Dict[str, int] = defaultdict(int)
    for cid, score, ranks in fused:
        row = rows.get(cid)
        if row is None:
            continue
        key = row["corpus"] + ":" + row["rel"]
        if per_file[key] >= max_per_file:
            continue
        per_file[key] += 1
        hits.append({
            "corpus": row["corpus"],
            "rel": row["rel"],
            "path": row["abs_path"],
            "ref": key,
            "heading": row["heading"],
            "title": row["title"],
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "score": round(score, 5),
            "lex_rank": ranks[0],
            "vec_rank": ranks[1] if len(ranks) > 1 else None,
            "bm25": round(lex_scores[cid], 3) if cid in lex_scores else None,
            "cosine": round(vec_scores[cid], 4) if cid in vec_scores else None,
            "modified": datetime.fromtimestamp(row["mtime"]).strftime("%Y-%m-%d"),
            "snippet": make_snippet(row["body"], terms, snippet_chars),
        })
        if len(hits) >= k:
            break
    return {"query": query, "mode": used_mode, "lexical_backend": lexical_label, "note": note,
            "lexical_coverage": round(coverage, 2), "vec_weight": round(vec_weight, 2) if used_mode == "hybrid" else 0.0,
            "corpora": list(corpora) if corpora else "all", "k": k, "hits": hits, "timings": timings}


def format_hits(result: Dict[str, Any]) -> str:
    lines = []
    head = "mode=%s (%s)" % (result["mode"], result["lexical_backend"])
    if result["mode"] == "hybrid":
        head += " lexical-coverage=%.2f vec-weight=%.2f" % (result.get("lexical_coverage", 0), result.get("vec_weight", 0))
    if result.get("note"):
        head += " - " + result["note"]
    lines.append(head)
    if not result["hits"]:
        lines.append("no hits")
    for i, h in enumerate(result["hits"], 1):
        lanes = []
        if h.get("lex_rank"):
            lanes.append("lex#%d" % h["lex_rank"])
        if h.get("vec_rank"):
            lanes.append("vec#%d" % h["vec_rank"])
        lines.append("%d. %s  L%d-%d  [%s]  %s" % (i, h["ref"], h["line_start"], h["line_end"], " ".join(lanes), h["modified"]))
        if h.get("heading"):
            lines.append("   # " + h["heading"])
        lines.append("   " + h["snippet"])
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Get / recent / status
# ----------------------------------------------------------------------------

def get_section(spec: str, heading: Optional[str], cfg: Dict[str, Any], store: Store,
                max_chars: int = 20000) -> Dict[str, Any]:
    """Read the live file (not the index) so the text is current; the index only resolves the path."""
    corpora = resolve_corpora(cfg)
    rows = store.find_file(spec)
    if not rows:
        return {"error": "no indexed file matches %r (use corpus:rel, an absolute path, or a unique relative path)" % spec}
    if len(rows) > 1:
        return {"error": "ambiguous", "candidates": ["%s:%s" % (r["corpus"], r["rel"]) for r in rows[:20]]}
    row = rows[0]
    path = Path(row["path"])
    roots = []
    for c in corpora:
        if c.root is not None:
            try:
                roots.append(c.root.resolve())
            except OSError:
                roots.append(c.root)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if not any(str(resolved).startswith(str(r)) for r in roots):
        return {"error": "refusing to read outside the configured corpora: %s" % path}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": "cannot read %s: %s" % (path, exc)}
    out: Dict[str, Any] = {"ref": "%s:%s" % (row["corpus"], row["rel"]), "path": row["path"], "title": row["title"]}
    if heading:
        _meta, sections = split_sections(text)
        needle = heading.lower().strip()
        picked = [s for s in sections if needle in s.heading_path.lower() or needle in s.heading.lower()]
        if not picked:
            out["error"] = "no heading matches %r" % heading
            out["headings"] = [s.heading_path for s in sections if s.level][:60]
            return out
        first = picked[0]
        # a heading's section includes its sub-sections: everything until the next heading at <= its level
        start = first.line_start
        end = first.line_end
        for s in sections:
            if s.line_start > first.line_start:
                if s.level and s.level <= first.level:
                    break
                end = s.line_end
        lines = text.splitlines()[start - 1:end]
        body = "\n".join(lines).strip()
        out.update({"heading": first.heading_path, "line_start": start, "line_end": end,
                    "matches": [s.heading_path for s in picked[:10]]})
    else:
        body = text
        out.update({"line_start": 1, "line_end": len(text.splitlines())})
    if len(body) > max_chars:
        out["truncated"] = True
        body = body[:max_chars] + "\n... [truncated at %d chars; read the file for the rest]" % max_chars
    out["text"] = body
    return out


def recent_files(cfg: Dict[str, Any], store: Store, corpus: Optional[str], days: float, limit: int) -> List[Dict[str, Any]]:
    out = []
    for r in store.recent(corpus, days, limit):
        out.append({"corpus": r["corpus"], "ref": "%s:%s" % (r["corpus"], r["rel"]), "path": r["path"],
                    "title": r["title"], "modified": datetime.fromtimestamp(r["mtime"]).strftime("%Y-%m-%d %H:%M"),
                    "chunks": r["chunks"], "size": r["size"]})
    return out


def config_hint(cfg: Dict[str, Any], corpora: List[Corpus]) -> str:
    """One line telling a reader what to do when nothing useful is configured; empty otherwise."""
    paths = cfg.get("_paths", {})
    if not corpora:
        return "no corpora configured: create %s (kb.py config --init) and list your notes folders" % paths.get("user", USER_CONFIG_PATH)
    if not any(c.available for c in corpora):
        return "no corpus folder exists on this machine: edit the corpora in %s" % (paths.get("loaded") or paths.get("user"))
    if paths.get("source") == "example":
        return "using config.example.json: copy it to %s (kb.py config --init) and edit the corpora" % paths.get("user")
    return ""


def status(cfg: Dict[str, Any], store: Store, embedder: Optional[Embedder], probe: bool = True) -> Dict[str, Any]:
    corpora = resolve_corpora(cfg)
    stats = store.corpus_stats()
    sig = embedder.signature if embedder else (store.get_meta("embed_sig") or "")
    done, total = store.embedding_coverage(sig)
    age = index_age_seconds(store)
    try:
        size = store.db_path.stat().st_size
        for suffix in ("-wal", "-shm"):
            extra = Path(str(store.db_path) + suffix)
            if extra.exists():
                size += extra.stat().st_size
    except OSError:
        size = 0
    out: Dict[str, Any] = {
        "version": __version__,
        "index_path": str(store.db_path),
        "db_bytes": size,
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "fts5_available": store.has_fts5,
        "lexical_backend": store.backend,
        "config": cfg.get("_paths", {}),
        "cli": str(HERE / "kb.py"),
        "files": sum(s.get("files", 0) for s in stats.values()),
        "chunks": total,
        "embeddings": {"signature": sig, "embedded": done, "total": total,
                       "coverage": round(done / total, 3) if total else 0.0},
        "index_age_s": round(age, 1) if age is not None else None,
        "last_index_report": json.loads(store.get_meta("last_index_report") or "{}"),
        "hint": config_hint(cfg, corpora),
        "corpora": [],
    }
    if embedder is not None:
        ok, reason = embedder.available(force=probe) if probe else embedder.available()
        out["embeddings"].update({"enabled": embedder.enabled, "url": embedder.url, "model": embedder.model,
                                  "reachable": ok, "reason": reason})
    for c in corpora:
        d = c.as_dict()
        d.update(stats.get(c.name, {}))
        out["corpora"].append(d)
    return out


def format_status(st: Dict[str, Any]) -> str:
    lines = ["kb-mcp %s  index=%s (%.1f MB)  python %s  sqlite %s  lexical=%s%s" % (
        st["version"], st["index_path"], st["db_bytes"] / 1e6, st["python"], st["sqlite"], st["lexical_backend"],
        "" if st["fts5_available"] else " (FTS5 missing)")]
    emb = st["embeddings"]
    if emb.get("enabled") is False:
        lines.append("embeddings: disabled (lexical search only)")
    else:
        lines.append("embeddings: %d/%d chunks (%s) via %s %s - %s" % (
            emb["embedded"], emb["total"], emb.get("signature", ""), emb.get("url", "?"),
            "reachable" if emb.get("reachable") else "UNREACHABLE", emb.get("reason", "")))
    age = st.get("index_age_s")
    lines.append("index: %d files, %d chunks; last run %s" % (
        st["files"], st["chunks"], ("%.0f min ago" % (age / 60)) if age is not None else "never"))
    cfg = st.get("config", {})
    lines.append("config: %s (%s)" % (cfg.get("loaded") or "built-in defaults", cfg.get("source", "")))
    for c in st["corpora"]:
        flag = "" if c["available"] else ("  (disabled)" if not c.get("enabled", True) else "  (folder not found)")
        lines.append("  %-16s files=%-4d chunks=%-5d embedded=%-5d  %s%s" % (
            c["name"], c.get("files", 0), c.get("chunks", 0), c.get("embedded", 0), c.get("root") or c["configured_path"], flag))
    if st.get("hint"):
        lines.append("hint: " + st["hint"])
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def setup_logging(verbose: bool, log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        try:
            path = Path(expand_user_path(log_file))
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(str(path), encoding="utf-8"))
        except OSError:
            pass
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def open_store(cfg: Dict[str, Any]) -> Store:
    return Store(Path(expand_user_path(cfg["index_path"])), str(cfg.get("lexical", {}).get("backend", "auto")))


def init_user_config(cfg: Dict[str, Any]) -> Tuple[Path, bool]:
    """Copy config.example.json to the user config path unless a file is there. (path, created)."""
    target = Path(cfg.get("_paths", {}).get("user") or USER_CONFIG_PATH)
    if target.exists():
        return target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    example = Path(cfg.get("_paths", {}).get("example") or EXAMPLE_CONFIG_PATH)
    if example.is_file():
        shutil.copyfile(str(example), str(target))
    else:
        target.write_text(json.dumps({k: v for k, v in BUILTIN_CONFIG.items()}, indent=2) + "\n", encoding="utf-8")
    return target, True


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="kb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--config", help="extra config file (merged last)")
    # the same two options after the subcommand; SUPPRESS keeps them from clobbering the values above
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--config", default=argparse.SUPPRESS, help="extra config file (merged last)")
    sub = ap.add_subparsers(dest="cmd", parser_class=lambda **kw: argparse.ArgumentParser(parents=[common], **kw))

    p = sub.add_parser("index", help="incremental index (sha256 per file; unchanged files skipped)")
    p.add_argument("--full", action="store_true", help="re-chunk every file")
    p.add_argument("--corpus", action="append", help="limit to a corpus (repeatable)")
    p.add_argument("--no-embed", action="store_true", help="skip embeddings this run")
    p.add_argument("--if-older-than", type=float, metavar="MIN", help="do nothing when the last run is newer than MIN minutes")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("search", help="hybrid search")
    p.add_argument("query")
    p.add_argument("--corpus", action="append")
    p.add_argument("-k", type=int)
    p.add_argument("--mode", choices=["auto", "lexical", "hybrid"], default="auto")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("get", help="section text by path (+ heading)")
    p.add_argument("path", help="corpus:rel, absolute path, or unique relative path")
    p.add_argument("--heading")
    p.add_argument("--max-chars", type=int, default=20000)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("recent", help="recently modified indexed files")
    p.add_argument("--corpus")
    p.add_argument("--days", type=float, default=7)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("status", help="index and corpora status")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-probe", action="store_true", help="do not contact the embedder")

    p = sub.add_parser("config", help="print the effective configuration")
    p.add_argument("--init", action="store_true", help="create the user config from config.example.json if missing")

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 2
    cfg = load_config()
    if args.config:
        cfg = deep_merge(cfg, _read_json(Path(args.config)))
    setup_logging(args.verbose, None if args.cmd in ("search", "get", "recent", "status", "config") else cfg.get("log_file"))

    if args.cmd == "config":
        if args.init:
            target, created = init_user_config(cfg)
            print(("created %s from config.example.json; edit its corpora" if created else "%s already exists; left unchanged") % target)
            return 0
        print(json.dumps(cfg, indent=2, default=str))
        return 0

    store = open_store(cfg)
    embedder = Embedder(cfg)
    try:
        if args.cmd == "index":
            if args.if_older_than is not None:
                age = index_age_seconds(store)
                if age is not None and age < args.if_older_than * 60:
                    if not args.quiet:
                        print("index is %.0f min old; nothing to do" % (age / 60))
                    return 0

            def progress(msg: str) -> None:
                if args.verbose:
                    log.info(msg)
            report = index(cfg, store, embedder, full=args.full, corpora=args.corpus, embed=not args.no_embed, progress=progress)
            if args.json:
                print(json.dumps(report, indent=2))
            elif not args.quiet:
                if report.get("skipped"):
                    print("skipped: %s" % report.get("reason"))
                    return 0
                if report["embedding_mode"] == "disabled":
                    emb_text = "disabled (lexical search only)"
                else:
                    emb_text = "%s (new %d, reused %d, pending %d%s)%s" % (
                        report["embedding_mode"], report["embeddings_new"], report["embeddings_reused"],
                        report["embeddings_pending"],
                        (", %.1fs" % report["embedding_seconds"]) if "embedding_seconds" in report else "",
                        (" - " + report["embedding_reason"]) if report.get("embedding_reason") and report["embedding_mode"] != "ollama" else "")
                print("indexed %d files (%d unchanged, %d removed), %d chunks written, %.1fs; embeddings: %s" % (
                    report["files_indexed"], report["files_unchanged"], report["files_removed"], report["chunks_written"],
                    report["seconds"], emb_text))
                for name, c in report["corpora"].items():
                    print("  %-16s files=%-4d indexed=%-4d unchanged=%-4d removed=%-3d chunks=%-5d %s" % (
                        name, c.get("files", 0), c.get("indexed", 0), c.get("unchanged", 0), c.get("removed", 0), c.get("chunks", 0), c.get("note", "")))
                hint = config_hint(cfg, resolve_corpora(cfg))
                if hint:
                    print("hint: " + hint)
            return 0
        if args.cmd == "search":
            result = search(args.query, cfg, store, embedder, k=args.k, corpora=args.corpus, mode=args.mode)
            print(json.dumps(result, indent=2) if args.json else format_hits(result))
            return 0
        if args.cmd == "get":
            out = get_section(args.path, args.heading, cfg, store, args.max_chars)
            if args.json:
                print(json.dumps(out, indent=2))
            elif "error" in out:
                print("error: %s" % out["error"])
                for c in out.get("candidates", []) or out.get("headings", []):
                    print("  " + c)
                return 1
            else:
                print("%s  L%d-%d%s" % (out["ref"], out["line_start"], out["line_end"], ("  # " + out["heading"]) if out.get("heading") else ""))
                print(out["text"])
            return 0
        if args.cmd == "recent":
            rows = recent_files(cfg, store, args.corpus, args.days, args.limit)
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                for r in rows:
                    print("%s  %-9s %s  (%s, %d chunks)" % (r["modified"], r["corpus"], r["ref"], r["title"], r["chunks"]))
            return 0
        if args.cmd == "status":
            st = status(cfg, store, embedder, probe=not args.no_probe)
            print(json.dumps(st, indent=2, default=str) if args.json else format_status(st))
            return 0
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
