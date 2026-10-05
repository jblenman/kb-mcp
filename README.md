# kb-mcp

A local, private knowledge-base MCP server for Claude Code over any folders of Markdown (and text) notes.

Point it at your notes — a notes repository, an Obsidian vault, Claude Code's auto-memory folders, a single journal file — and a Claude Code session asks one question and gets the right note back: `kb_search("restore test FAT32")` → `notes:tools/backup.md › Backup routine › Restore test`, lines 6–9, instead of a round of Greps and Reads. The index is one SQLite file under your home folder. Nothing leaves the machine unless you point the optional embedder at another host.

- **Stdlib Python 3.8+**, no dependencies, one process. Tested on 3.9 and 3.13.
- **Retrieval**: BM25 over SQLite FTS5 (pure-Python fallback when the interpreter lacks FTS5), optionally fused with embeddings from a local Ollama model by adaptive reciprocal rank fusion. Without an embedder it is a fast lexical search and says so.
- **Chunks by heading**, with the heading path and exact line range kept, so a hit is a place in a file, not a blob.
- **Incremental and self-refreshing**: unchanged files are skipped by hash; the MCP server re-indexes in the background when the index is older than 30 minutes.
- **Five tools**: `kb_search`, `kb_get`, `kb_recent`, `kb_reindex`, `kb_status`; the same operations as a CLI.
- Installable three ways: a plain `claude mcp add`, as a Claude Code plugin from this repository, or from another marketplace.

## Quick start

```bash
git clone https://github.com/jblenman/kb-mcp ~/kb-mcp
python3 ~/kb-mcp/kb_mcp/kb.py config --init        # writes ~/.claude/kb-mcp/config.json from config.example.json
$EDITOR ~/.claude/kb-mcp/config.json               # list your notes folders under "corpora"
python3 ~/kb-mcp/kb_mcp/kb.py index                # first build (seconds; a minute with embeddings)
python3 ~/kb-mcp/kb_mcp/kb.py search "what I decided about backups"
claude mcp add --scope user kb-mcp -- python3 ~/kb-mcp/kb_mcp/mcp_server.py
```

Then in any Claude Code session: `kb_search`, `kb_get`, `kb_recent`, `kb_reindex`, `kb_status` are available (as `mcp__kb-mcp__kb_search` etc.).

Windows: use `py` (or the full path of a Python launcher, see [Windows notes](#windows-notes)) in place of `python3`, and `%USERPROFILE%\kb-mcp` in place of `~/kb-mcp`.

## Install

### 1. Plain MCP registration (`claude mcp add`)

Registers the server for your user, no plugin involved. Tool names are `mcp__kb-mcp__<tool>`.

```bash
claude mcp add --scope user kb-mcp -- python3 /absolute/path/kb-mcp/kb_mcp/mcp_server.py
claude mcp add --scope user kb-mcp -- py C:/Users/<you>/kb-mcp/kb_mcp/mcp_server.py        # Windows
```

Check with `claude mcp list` (the server shows `✓ Connected` once it answers the handshake). Remove with `claude mcp remove kb-mcp`.

### 2. As a plugin from this repository

The repository is a Claude Code plugin (`.claude-plugin/plugin.json` with the MCP server declared inline, plus a `kb` skill) and also a one-entry marketplace, so it installs directly:

```bash
claude plugin marketplace add jblenman/kb-mcp        # or the path of a local clone
claude plugin install kb-mcp@kb-mcp --scope user
```

The server then appears as `plugin:kb-mcp:kb`; tool names are `mcp__plugin_kb-mcp_kb__<tool>`. The skill `kb` is loaded with it (about 190 always-on tokens per session, per `claude plugin details`).

- A marketplace added from a **local path** loads the plugin **in place** from that clone: `git pull` is the whole update. A marketplace added from a **GitHub or git URL** copies the plugin into `~/.claude/plugins/cache/`; `claude plugin update kb-mcp@kb-mcp` picks up new commits. (Verified on Claude Code 2.1.289: `claude plugin list` prints `Read from: <clone>` for the local-path case.)
- Session-only trial, nothing installed: `claude --plugin-dir /path/to/kb-mcp`.
- Interpreter: the plugin's server entry (`plugin.json` → `mcpServers`) starts the server with `${KIT_PYTHON:-python3}`. Where `python3` is not on `PATH` (Windows), set `"env": {"KIT_PYTHON": "<interpreter>"}` in `~/.claude/settings.json` — a settings `env` value reaches a plugin's `.mcp.json` command. See [Windows notes](#windows-notes).

### 3. From another marketplace

A marketplace can list this repository as a GitHub-sourced plugin; its users then run `claude plugin install kb-mcp@<that-marketplace>`. The entry (fields per Claude Code's marketplace reference; only the entry's own fields are shown to users before install):

```json
{
  "name": "kb-mcp",
  "source": { "source": "github", "repo": "jblenman/kb-mcp" },
  "description": "A local, private knowledge-base MCP server over your own folders of Markdown and text notes: kb_search, kb_get, kb_recent, kb_reindex, kb_status over a SQLite index (BM25 via FTS5, optional Ollama embeddings, adaptive rank fusion) that refreshes itself. Stdlib Python 3.8+.",
  "version": "1.0.0",
  "homepage": "https://github.com/jblenman/kb-mcp",
  "license": "MIT",
  "category": "knowledge",
  "tags": ["knowledge-base", "notes", "search", "rag", "mcp", "markdown", "obsidian"]
}
```

Add `"ref": "v1.0.0"` (and a full 40-character `"sha"`) to pin a release.

## Configuration

Resolution order, first file found wins and is merged over the built-in defaults:

1. `$KB_MCP_CONFIG` (a path)
2. `~/.claude/kb-mcp/config.json`
3. `config.example.json` beside the package (the shipped example: a `~/notes` corpus plus two disabled examples)

`python3 kb_mcp/kb.py config` prints the effective configuration and which file it came from; `config --init` copies the example to `~/.claude/kb-mcp/config.json`. `$KB_MCP_INDEX` overrides the index path. Keys starting with `_` are comments. You may delete every key you do not change.

```json
{
  "corpora": [
    {"name": "notes", "path": "~/notes", "include": ["**/*.md", "**/*.txt"], "exclude": ["drafts/**"], "description": "my notes"},
    {"name": "vault", "path": "~/Documents/Obsidian/Vault", "include": ["**/*.md"], "exclude": ["templates/**"]},
    {"name": "claude-memory", "path": "~/.claude/projects", "include": ["*/memory/**/*.md"], "description": "Claude Code auto-memory"},
    {"name": "journal", "path": "~/Documents/journal.md"}
  ],
  "exclude_globs": ["**/.git/**", "**/node_modules/**", "**/.obsidian/**", "**/.trash/**"],
  "max_file_bytes": 2000000,
  "index_path": "~/.claude/kb-mcp/index.sqlite",
  "log_file": "~/.claude/kb-mcp/kb-mcp.log",
  "embedder": {"enabled": true, "url": "http://127.0.0.1:11434", "model": "nomic-embed-text"},
  "chunking": {"min_words": 60, "target_words": 500, "max_words": 800},
  "lexical": {"backend": "auto"},
  "search": {"k": 6, "pool": 40, "vec_pool": 20, "rrf_k": 20, "vec_weight_min": 0.2, "vec_weight_max": 1.0, "snippet_chars": 600, "max_per_file": 3},
  "freshness": {"auto_index_minutes": 30}
}
```

**Corpora.** Each entry is a folder of notes. `name` (letters, digits, `.`, `_`, `-`) prefixes every result as `name:relative/path`. `path` takes `~` and environment variables; it may also be a **single file**, which becomes a one-file corpus (a journal or a running notes file is just another corpus). `include` globs are relative to the folder (default `**/*.md`, `**/*.markdown`, `**/*.txt`); `exclude` globs skip files; `enabled: false` keeps an entry but skips it. A corpus whose folder does not exist on this machine is skipped with a note in `kb_status`, so one config can serve several machines. A corpus removed from the config is purged from the index at the next run.

Corpus examples:

| Notes | Entry |
|---|---|
| A notes repository | `{"name": "notes", "path": "~/notes", "include": ["**/*.md"]}` |
| An Obsidian vault | `{"name": "vault", "path": "~/Documents/Obsidian/Vault", "include": ["**/*.md"], "exclude": ["templates/**"]}` (the `.obsidian/` and `.trash/` folders are excluded by default) |
| Claude Code auto-memory (one folder per project) | `{"name": "claude-memory", "path": "~/.claude/projects", "include": ["*/memory/**/*.md"]}` — frontmatter `description` becomes the chunk heading when a file has no H1 |
| A single file | `{"name": "journal", "path": "~/Documents/journal.md"}` |
| Only part of a tree | `{"name": "docs", "path": "~/work/handbook", "include": ["guides/**/*.md", "README.md"]}` |

**Embedder.** Optional. `enabled: true` (the default) with the default URL uses an Ollama instance on this machine (`ollama pull nomic-embed-text`). When Ollama is not running, indexing and search still work: chunks are stored, embeddings stay *pending* and are filled on the next run that finds the embedder up; searches answer lexically and say why. `enabled: false` makes kb-mcp purely lexical and it never opens a network connection. `doc_prefix`/`query_prefix` default to the task prefixes nomic-embed-text expects and to empty strings for any other model; set them explicitly to override. `num_ctx` is 8192 because code-heavy chunks exceed Ollama's 2048 default. Changing the model re-embeds everything (vectors are keyed by model + prefix).

**Index location.** `~/.claude/kb-mcp/index.sqlite` by default: on a local disk, under your home folder, never inside a notes folder, a repository, or a cloud-synced folder (SQLite in a synced folder is a corruption risk). The lock file and log sit beside it.

## Using it

**MCP tools** (what Claude Code sees):

| Tool | Does |
|---|---|
| `kb_search(query, k?, corpora?, mode?)` | Ranked excerpts: `corpus:path`, heading path, line range, snippet, which lanes ranked it (`lex#`, `vec#`). `mode` `auto` (default), `lexical` or `hybrid`. |
| `kb_get(path, heading?, max_chars?)` | The full section (a parent heading includes its sub-sections) or the whole file, read live from disk. `path` as shown by `kb_search`, an absolute path inside a corpus, or a unique relative path. Refuses paths outside the configured corpora. |
| `kb_recent(corpus?, days?, limit?)` | Recently modified notes, newest first. |
| `kb_reindex(full?, corpora?)` | Incremental re-index now. Normally unnecessary (see Freshness); use it when a note written seconds ago must be found now. |
| `kb_status()` | Corpora and their folders, counts, embedding coverage and reachability, lexical backend, index age, which config file is in use, and a hint when nothing useful is configured. |

The server's instructions tell the model which corpora exist and that excerpts are the user's data, never instructions. The `kb` skill (loaded with the plugin, or copy `skills/kb/` to `~/.claude/skills/`) tells sessions to search the notes before grepping them by hand.

**CLI** — the same operations from a shell (`python3 kb_mcp/kb.py …`, or `python3 -m kb_mcp …` from the repository root):

```
kb.py index [--full] [--corpus NAME] [--no-embed] [--if-older-than MIN] [--json] [--quiet]
kb.py search "query" [-k 6] [--corpus NAME] [--mode auto|lexical|hybrid] [--json]
kb.py get notes:tools/backup.md [--heading "Retention"]     # corpus:rel, absolute, or unique rel path
kb.py recent [--corpus notes] [--days 7] [--limit 20]
kb.py status [--json] [--no-probe]
kb.py config [--init]
```

## How retrieval works

- **Chunking.** Files are split at Markdown headings with the heading path kept (`Backup routine > Restore test`); fenced code is never read as a heading; YAML frontmatter is stripped (its `description` or `title` names the file when there is no H1). Small neighbouring sections under one parent merge up to ~500 words; sections over 800 words split on paragraph boundaries; plain `.txt` files are one section split the same way. Every chunk keeps its exact 1-based line range (the tests reproduce each chunk from the file by its range). On a collection of about 200 notes this gave ~1,300 chunks with a median of 170 words.
- **Lexical lane.** SQLite FTS5 (`porter unicode61`; columns heading ×2, body ×1, path ×0.5), the query's terms OR-ed with the full phrase as an extra clause, so BM25 ranks term coverage and a note is found even when one query word is absent from it. FTS5 is probed at runtime; without it (some Windows builds) the same postings go into a plain table and BM25 runs in Python — same top-3 on the spot-check queries, about 2.5× slower to index.
- **Vector lane.** Embeddings from Ollama, L2-normalised float32 blobs in SQLite; cosine = dot product over `array('f')` in pure Python, cached in the server process (28 ms for 1,300 vectors). An input the model rejects at every length cap is left pending and reported, never fatal.
- **Fusion.** Weighted reciprocal rank fusion (k = 20, vector pool 20). The vector weight adapts to how much of the query the lexical top hit covers: 1.0 when the words are not in the notes (a paraphrase), down to 0.2 when every term is (an identifier query). Measured on 20 queries with known answers (14 identifier, 6 paraphrase) over that 200-note collection: lexical only p@3 0.57 / MRR 0.84, vector only 0.52 / 0.67, fixed RRF k = 60 0.60 / 0.79, adaptive 0.62 / 0.89. Each result prints the mode, coverage and weight so the behaviour is inspectable.
- **Degradation.** `kb_status` probes the embedder (2 s, cached 30 s). Down at index time: embeddings pending, filled later. Down at query time: lexical answer with the reason in the header (`mode=lexical (fts5) - embedder unreachable at …`). Disabled: no probe, no network, no note.
- **Incremental.** Per file `mtime+size` pre-check, then sha256; unchanged files are skipped, removed files purged. Within a changed file, chunks whose text is unchanged keep their embeddings (a full re-chunk of 1,300 chunks re-embedded 35). WAL mode, so searches never wait on an index run.

Measured on a 2021 Apple-silicon laptop (Python 3.13, Ollama on another machine over a LAN), ~200 files / ~1,300 chunks: full build 57 s, of which 0.3 s chunking and the rest embedding at ~23 chunks/s; incremental run with nothing changed 0.13 s; three changed files 0.7 s; lexical search 1–3 ms; hybrid search 60–65 ms (query embedding + cosine scan); index 13 MB. The 4-file sample in [Verify it works](#verify-it-works) indexes in 0.1 s.

## Freshness: the index keeps itself current

The MCP server does it. When the last index run is older than `freshness.auto_index_minutes` (30), it runs an incremental index in a background thread at start (after `initialized`) and again before a search or `kb_recent`; the current index answers meanwhile and the result footer says `refreshing in the background`. `kb_reindex` exists for "I just wrote the note, find it now". `KB_MCP_NO_AUTOINDEX=1` in the server's environment disables the self-refresh.

**At exit** (stdin closed, or the first SIGINT/SIGTERM) the server tells a running index to stop at its next file or embedding batch and waits up to 3 s for it to commit and release the lock. A stopped run purges nothing (its walk was partial) and does not mark the index fresh, so the next start finishes the job; work already done is kept, so a first build spread over several short `claude -p` runs still completes. Claude Code ends a stdio server with SIGINT, SIGTERM about 0.1 s later and SIGKILL about 0.5 s after the first signal, without closing stdin (measured on CLI 2.1.289, macOS); the server treats the first signal as a normal exit and ignores the later ones, which leaves time to finish an in-flight embedding call.

A SessionStart hook is optional and redundant with the server's own refresh; if you want one anyway: `python3 /path/kb-mcp/kb_mcp/kb.py index --if-older-than 30 --quiet || exit 1` (a no-op while fresh).

Only WARNING and above go to stderr (Claude Code records every stderr line of an MCP server as an `[ERROR]`); INFO goes to the log file.

## The lock

One index writer per database: `index.lock` beside the index, created with `O_EXCL`, holding `{"pid", "created"}`.

- A lock whose owner PID is **no longer running** is taken over at once (so a server killed outright, SIGKILL or TerminateProcess, blocks nobody). On Windows, where PIDs are reused quickly, a process with that PID that **started after the lock was created** is also not its owner (checked with `GetProcessTimes`; PIDs above the DWORD range are treated as corrupt records, never passed to the API). POSIX hands PIDs out in sequence, so a reused PID there is left to the age limit.
- Any lock **older than 15 minutes** is taken over (a hung owner).
- A lock whose owner **cannot be checked**, or whose record is not written yet or unreadable, is respected until the age limit.
- Takeover renames the lock aside first (atomic), compares the bytes, and puts back a lock that is not the one judged; release compares bytes too and leaves a lock another run took over in the meantime. So of several runs judging the same stale lock exactly one removes it, and nobody removes a live one.

Why it matters: before this design, a server killed mid-index left a lock that blocked every refresh and `kb_reindex` for 15 minutes, so a machine without an index searched an empty one.

## Privacy

- The index, lock and log live under `~/.claude/kb-mcp/` (configurable), never in a notes folder or a repository. Nothing is written into the corpora.
- With the embedder disabled, or pointed at `127.0.0.1`, kb-mcp opens no network connection. With the embedder enabled, the **text of every chunk** and of every query is sent to the configured Ollama URL; keep that on your own machine or your own network.
- `kb_get` refuses to read files outside the configured corpora, even when asked with an absolute path.
- Requests to the embedder pass through two no-op hooks in `kb.py` (`pace_request`, `report_response`); if you ever point the embedder at a shared or public host, that is where request spacing or a back-off would go.

## Headless use and permission rules

A `claude -p` run calls MCP tools only with an allow rule that names them; interactive sessions ask otherwise. The rule depends on how the server was registered, because tool names differ:

| Registration | Server name | Tool names | Allow rule (whole server) |
|---|---|---|---|
| `claude mcp add … kb-mcp …` | `kb-mcp` | `mcp__kb-mcp__kb_search` … | `mcp__kb-mcp` |
| plugin `kb-mcp` (server key `kb` in `plugin.json`) | `plugin:kb-mcp:kb` | `mcp__plugin_kb-mcp_kb__kb_search` … | `mcp__plugin_kb-mcp_kb` |

Single tools work too (`mcp__kb-mcp__kb_search mcp__kb-mcp__kb_status`). In `~/.claude/settings.json` the same strings go into `permissions.allow`. `--allowedTools` is variadic and swallows a trailing prompt, so put the prompt before it. Verified on CLI 2.1.289 with Haiku: both registrations, `kb_status` then `kb_search`, the top hit quoted correctly, about $0.06 and 6–7 s per run; the MCP tools were deferred, so the model's first turn was a `ToolSearch` that loaded them, which costs nothing to you.

To test a server without touching your MCP configuration: `--mcp-config <file> --strict-mcp-config` with a one-server `mcpServers` object (see [Verify it works](#verify-it-works)). `--strict-mcp-config` also leaves out every **plugin** MCP server (verified: with five plugins installed, the init event listed only the dynamic server; the plugins' skills still loaded) and the claude.ai connectors, so it is the right flag for a clean proof.

## Verify it works

Copy-paste proof with a throwaway corpus and the embedder disabled (no Ollama needed). On Windows run it in Git Bash, or translate the paths.

```bash
P=$(mktemp -d); mkdir -p "$P/notes/tools"
printf '# Backup routine\n\nWeekly backups go to the external drive.\n\n## Restore test\n\nRestore one folder each quarter. The archive stopped at 4 GB on a FAT32 volume; use exFAT.\n' > "$P/notes/tools/backup.md"
printf '# Printer notes\n\n## Nozzle clogs\n\nA clogged nozzle shows as thin lines. Do a cold pull before replacing the nozzle.\n' > "$P/notes/tools/printer.md"
printf '{"corpora":[{"name":"notes","path":"%s/notes"}],"index_path":"%s/index.sqlite","log_file":"%s/kb-mcp.log","embedder":{"enabled":false}}\n' "$P" "$P" "$P" > "$P/config.json"
export KB_MCP_CONFIG="$P/config.json"
python3 kb_mcp/kb.py index                          # indexed 2 files ..., embeddings: disabled (lexical search only)
python3 kb_mcp/kb.py search "clogged nozzle" -k 2   # 1. notes:tools/printer.md  L1-5 ... # Printer notes > Nozzle clogs
python3 kb_mcp/kb.py get notes:tools/backup.md --heading "Restore test"
python3 kb_mcp/kb.py status                         # lexical=fts5 (or python), config: .../config.json (env)
```

Through Claude Code, with the server registered only for this run (nothing is written to your settings):

```bash
printf '{"mcpServers":{"kb-mcp":{"command":"python3","args":["%s/kb_mcp/mcp_server.py"],"env":{"KB_MCP_CONFIG":"%s/config.json"}}}}\n' "$PWD" "$P" > "$P/mcp.json"
claude -p "Call kb_status, then kb_search for 'clogged nozzle'. Reply with the file count and the top hit's corpus:path." \
  --model haiku --mcp-config "$P/mcp.json" --strict-mcp-config --allowedTools mcp__kb-mcp < /dev/null
```

Expected: a two-line answer naming 2 files and `notes:tools/printer.md`. Add `--output-format stream-json --verbose` to see the init event (`mcp_servers: [{"name": "kb-mcp", "status": "connected"}]`) and each tool call.

Installed as a plugin: `claude plugin list` shows `kb-mcp@kb-mcp … enabled`; in a session, `kb_status` answers (or `/mcp` lists `plugin:kb-mcp:kb`).

## Windows notes

- **Interpreter.** There is no `python3` on `PATH` on a stock Windows Python. For `claude mcp add`, use `py` (the python.org launcher) or the full path of an interpreter. For the plugin, set the interpreter once in `~/.claude/settings.json`: `"env": {"KIT_PYTHON": "C:/Windows/py.exe"}` — settings `env` values reach the plugin's server command (`${KIT_PYTHON:-python3}`). Forward slashes work everywhere Claude Code reads a path.
- **Store / PyManager installs.** When `py`, `python3` and `python` are **app-execution aliases** under `%LOCALAPPDATA%\Microsoft\WindowsApps` (a Store or Python-install-manager install; `(Get-Item (where.exe py)).Attributes` shows `ReparsePoint`), the alias can be denied to other logon sessions — observed on Windows 11: from an SSH logon the aliases answered "Access is denied" while a desktop session used them. Do not put an alias into an MCP command or a hook. Use the launcher executable, a regular file that survives Python upgrades: `C:/Users/<you>/AppData/Local/Python/bin/python.exe` for a PyManager install, `C:/Windows/py.exe` for a classic python.org install. Never a versioned path such as `Python314/python.exe`.
- **FTS5** is probed at runtime (`kb.py status` prints `fts5_available`); the Python BM25 fallback is automatic. Check with `py -c "import sqlite3; sqlite3.connect(':memory:').execute('create virtual table t using fts5(x)')"`.
- **Encoding.** The CLI switches stdout to UTF-8; the MCP server talks bytes on stdin/stdout, so the console code page does not matter; the plugin's server entry sets `PYTHONIOENCODING`/`PYTHONUTF8` anyway. For `claude mcp add` you may add `-e PYTHONUTF8=1`.
- **Paths** are stored posix-style with the drive letter (`C:/Users/<you>/notes/a.md`); the database is per machine, so nothing needs to be portable. Keep `index_path` on a local disk, not a synced folder.
- **Signals.** Windows has no SIGINT/SIGTERM for a stdio child; a killed server leaves its lock to the PID check, which takes it over at the next run (the Windows-only test covers reused PIDs).
- The suite's Windows-only test runs there; the POSIX signal test is skipped on Windows.

## Limits

- Markdown and plain text only; no PDF, HTML or Office files. Files over `max_file_bytes` (2 MB) are skipped. A symlinked **file** that matches a glob is indexed; `**` does not descend into symlinked **folders** (checked on Python 3.9 and 3.13), so link the files you want, not a folder.
- One machine, one index: the index is not shared and cannot be queried over the network. Point several machines at the same notes (a git clone each) and each builds its own index in seconds.
- `kb_get` reads the live file, so a hit from a stale index can show a different line range than the current file; the refresh fixes it within the freshness window.
- The embedder is Ollama's `/api/embed` only (no OpenAI-compatible endpoints). Hybrid search needs the embedder reachable at **query** time as well as at index time.
- Heading detection is ATX (`#`) only; Setext underlines are not headings. The frontmatter parser reads simple `key: value` lines.
- The vector scan is linear and in Python: fine to a few thousand chunks (28 ms at 1,300), not meant for hundreds of thousands.
- The server trusts its config file; a corpus pointed at a sensitive folder indexes it.

## Rollback

- Plain registration: `claude mcp remove kb-mcp` (add `--scope user` if it was added there).
- Plugin: `claude plugin uninstall kb-mcp@kb-mcp`, then `claude plugin marketplace remove kb-mcp` (verified: both leave `enabledPlugins` and `extraKnownMarketplaces` empty).
- Data: delete `~/.claude/kb-mcp/` (index, lock, log, config). Nothing else is written anywhere.

## Tests

```bash
python3 tests/test_kb_mcp.py          # 41 tests; passes on Python 3.13 and 3.9 (one Windows-only test skipped elsewhere)
```

Temporary fixture corpus, a fake embedder or a stub Ollama on 127.0.0.1 (no network): chunking with exact line ranges, fences, frontmatter, plain text; config resolution order and corpus forms; incremental skip / change / delete / corpus removal; lexical and hybrid search with adaptive weighting; embedder down, disabled, and recovering; the Python BM25 backend and a backend switch; `get` / `recent` / `status`; the MCP handshake (protocol echo and fallback, `server/discover`), `tools/list`, every tool via `tools/call`; the CLI and the stdio server as subprocesses; the lock (dead owner taken over at once, live owner respected, unwritten record respected until the age limit, takeover puts back a lock taken in between, release leaves a lock that is not ours, reused PID on Windows); cooperative stop (purges nothing, stays stale); the real server process during a background index (stdin closed → lock released and INFO kept out of stderr; killed → the next run takes the lock over; SIGINT then SIGTERM → exit 0 with the lock released).

## License

MIT — see [LICENSE](LICENSE).
