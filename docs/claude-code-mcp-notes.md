# Claude Code and stdio MCP servers — verified mechanics

What was learned building and running kb-mcp against Claude Code. Each item says which CLI version it was checked on; the behaviour may change in later releases. "Verified" means observed in a run, not read in the docs.

## Protocol and handshake

- **Claude Code 2.1.286+ asks for MCP protocol version `2025-11-25`.** kb-mcp echoes `2025-11-25`, `2025-06-18`, `2025-03-26` or `2024-11-05` and answers any other or missing version with `2025-06-18`. The CLI's connection log shows the result as `negotiatedProtocolVersion`. (2.1.286, 2.1.289)
- **2.1.289 probes `server/discover` before `initialize`.** A JSON-RPC `-32601 method not found` answer is fine; the handshake proceeds. (2.1.289)
- Transport is newline-delimited JSON-RPC 2.0 on stdin/stdout. The server must write nothing but JSON-RPC to stdout. `resources/list` and `prompts/list` may be answered with empty lists; unknown notifications are ignored.
- **Every stderr line of an MCP server is recorded as `[ERROR] MCP server "<name>" Server stderr: …`**, whatever its level, including an innocent "ready" line. Send only WARNING and above to stderr and INFO to a log file. (2.1.286)

## Shutdown

- **Claude Code ends a stdio server with SIGINT, SIGTERM about 0.1 s later, and SIGKILL about 0.5 s after the first signal, and never closes stdin.** A server with background work treats the first signal as a normal exit, ignores the later ones, and has well under half a second to commit and release any lock. Work that cannot stop in time must be safe to lose and its lock must be recoverable (kb-mcp's lock names the owner PID and is taken over once that PID is gone). (2.1.289, macOS)
- Windows has no such signals for a stdio child: the process is terminated outright. Design locks for that case.
- Headless `claude -p` runs are short: a first index build that needs a minute will be interrupted. Make long jobs resumable across runs (commit as you go, do not mark the job done until it is).

## Tool names and permission rules

- **Tool-name prefixes differ by registration path.** Registered with `claude mcp add <name> …`: `mcp__<name>__<tool>` (`mcp__kb-mcp__kb_search`). Loaded from a plugin: `mcp__plugin_<plugin>_<server-key>__<tool>` where the server key is the key in the plugin's `.mcp.json` (`mcp__plugin_kb-mcp_kb__kb_search`; the server lists as `plugin:kb-mcp:kb`). Permission rules and hook matchers must use the prefix of the path actually used. (2.1.289, both paths run)
- **Headless MCP calls need an allow rule.** `claude -p … --allowedTools mcp__kb-mcp` allows the whole server; `mcp__kb-mcp__kb_search mcp__kb-mcp__kb_status` allows single tools. Without a rule the call is refused as needing approval. Interactive sessions put the same strings in `permissions.allow`. (2.1.289)
- **`--allowedTools` is variadic and swallows a trailing prompt.** Put the prompt before it: `claude -p "…" --allowedTools mcp__kb-mcp --model haiku`.
- **MCP tools can be deferred.** In the runs above the model's first turn was a `ToolSearch` call that loaded the two tool schemas it needed, then the real calls followed; the init event still listed every tool name. Budget one extra turn in `--max-turns`. (2.1.289, Haiku)

## Testing a server without touching the user's configuration

- **`--mcp-config <file> --strict-mcp-config`** runs with only the servers in that file. Verified on 2.1.289: with five plugins installed (two of them with MCP servers) and four claude.ai connectors enabled, the `--output-format stream-json --verbose` init event listed `mcp_servers: [{"name": "kb-mcp", "status": "connected", "source": "dynamic"}]` and only `mcp__kb-mcp__*` tool names; the plugins' **skills** still loaded. Without `--strict-mcp-config` the same run listed the plugin servers (`source: "plugin"`) and the connectors (`source: "claudeai"`). So `--strict-mcp-config` excludes plugin MCP servers as well as user- and project-scope ones.
- **`claude --plugin-dir <path>`** loads a plugin for one session with nothing installed. `claude --plugin-dir <path> plugin details <name>` prints the component inventory and the always-on token cost (kb-mcp: ~186 tokens for its skill description; MCP tool schemas are "resolved at runtime; not counted"). (2.1.289)
- The init event of `--output-format stream-json --verbose` lists `mcp_servers`, `tools`, `skills`, `plugins` and `slash_commands` before any model call — a cheap way to check what a configuration loads.
- **`CLAUDE_CONFIG_DIR=<empty dir>`** isolates plugin state: `claude plugin marketplace add`, `install`, `list`, `uninstall`, `marketplace remove` all wrote only into that directory's `settings.json` (`extraKnownMarketplaces`, `enabledPlugins`) and left it empty again after the rollback. A session under such a dir is not logged in, so use it for install mechanics, not for model calls. (2.1.289)

## Plugins

- **A plugin added from a local-path marketplace loads in place from that directory**; `claude plugin list` prints `Read from: <that path>`. A marketplace added from a GitHub or git URL copies the plugin into `~/.claude/plugins/cache/`, and files outside the plugin directory are not copied — keep everything the server needs inside the plugin (kb-mcp's server, config example and skill all sit in the repository that is the plugin). (2.1.289 for the in-place observation)
- **A repository can be a plugin and a one-entry marketplace at once**: `.claude-plugin/plugin.json` plus `.claude-plugin/marketplace.json` with `"source": "."`. `claude plugin validate . --strict` then validates the marketplace manifest; validate the plugin manifest by its path (`claude plugin validate ./.claude-plugin/plugin.json --strict`). A display field set on the marketplace entry (e.g. `description`) is what users see, even when `plugin.json` differs, so keep the two identical. (2.1.289)
- **`${KIT_PYTHON:-python3}` as the plugin's server command works** (in a plugin's `.mcp.json` or inline under `mcpServers` in `plugin.json`), with no variable set (falls back to `python3`) and with `"env": {"KIT_PYTHON": "<path>"}` in `~/.claude/settings.json` — settings `env` values reach a plugin's `.mcp.json` command. The server process also inherits the parent's environment (a `KB_MCP_CONFIG` exported before `claude` was visible to the server). (2.1.289)
- `claude plugin validate --strict` treats warnings as errors; a missing `author` in `plugin.json` is a warning. Component paths stay inside the plugin; no symlinks (Windows git writes them as text); no `CLAUDE.md` at the plugin root (not loaded).

## Windows

- A stock Windows Python has no `python3` on `PATH`. Use `py` (python.org launcher) or a full interpreter path in `claude mcp add`, and `env.KIT_PYTHON` for the plugin.
- On a Store or Python-install-manager install, `py`/`python3`/`python` are app-execution aliases under `%LOCALAPPDATA%\Microsoft\WindowsApps`. Such an alias can be denied to other logon sessions (observed on Windows 11: "Access is denied" from an SSH logon while a desktop session used the same alias), and a hook or MCP command that starts with it then fails silently. Use the launcher executable instead: `C:/Users/<you>/AppData/Local/Python/bin/python.exe` (PyManager) or `C:/Windows/py.exe` (classic installer). `(Get-Item (where.exe py)).Attributes` showing `ReparsePoint` identifies an alias.
- Forward slashes work in every path Claude Code reads (`.mcp.json`, settings, hooks).
- **Declare a root-level plugin's server inline in `plugin.json` (`mcpServers`), not in a root `.mcp.json`.** When the plugin is the repository root, Claude Code also reads a root `.mcp.json` as a *project* server for anyone working inside a clone: it shows as pending approval with `Missing environment variables: CLAUDE_PLUGIN_ROOT`, and approving it gives a broken server. The inline form validates, loads through `--plugin-dir` and from a marketplace, and has no such side effect (checked on 2.1.289).
