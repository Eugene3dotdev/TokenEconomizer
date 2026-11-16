"""ZeroClaw (zeroclaw-labs/zeroclaw): a session read from its ACP session store.

ZeroClaw keeps the sessions its ACP server (`zeroclaw acp`) runs in SQLite, in
`<config dir>/data/sessions/acp-sessions.db` (`crates/zeroclaw-infra/src/acp_session_store.rs`,
0.9.0): a row per session in `acp_sessions` (`session_uuid`, `workspace_dir`, `created_at`,
`last_activity`, ...), its visible transcript in `acp_messages` (`role`, `content`), and each
tool call as two rows of `acp_tool_calls` on the assistant message that made it: `in`, whose
`payload` is the call's arguments as JSON, and `out`, whose `payload` is the result the model
read (cut at 16 KB). Every row of one turn carries that turn's `created_at`, an RFC 3339 time with
nanoseconds.

The config dir is `ZEROCLAW_CONFIG_DIR`, else the install `ZEROCLAW_DATA_DIR` (or the deprecated
`ZEROCLAW_WORKSPACE`) points at, else `~/.zeroclaw` (`crates/zeroclaw-config/src/schema.rs`).
Sessions of `zeroclaw agent` and of the channels are kept elsewhere (`sessions.db`), with no
directory to place them in a repository, and are not read.

A fold copies a session's rows into one JSON file beside the other state (`zeroclaw_export`), and
this module reads that file:

    {"info": {<columns of the acp_sessions row>},
     "messages": [{"role", "content", "created_at", "tools": [{tool_call_id, tool_name,
                   event_kind, payload}]}]}

The copy leaves out the words of the conversation: a prompt is `[ZeroClaw prompt]`, which is
enough to count it, and what the model said is not kept. A tool's arguments and result stay: they
are what is learned.

What the store does not say: the `out` row's `outcome` is always `unknown`, and ZeroClaw drops a
tool's success flag before the ACP boundary, so a failed call is told by the text ZeroClaw puts in
front of the result it gives the model (`FAILURE_PREFIXES`); a shell result carries no exit code.
No token usage is stored per session, so a session is unpriced.
"""
import json
import os
import pathlib
import sqlite3
from datetime import datetime

DB_PARTS = ("data", "sessions", "acp-sessions.db")
PROMPT = "[ZeroClaw prompt]"
# Rows ZeroClaw writes into a transcript that are not the person's: the note it puts where it
# trimmed earlier turns, and the marker of a turn whose stream was cut.
TRIM_BREADCRUMB = "[earlier turns omitted to fit the context window]"
INTERRUPTED_ROLE = "__zeroclaw_turn_stream_interrupted__"
# The prefix of every result ZeroClaw writes for a call that did not succeed
# (`crates/zeroclaw-runtime/src/agent/tool_execution.rs`, `turn/approval_gate.rs`,
# `turn/call_prep.rs`, `locales/en/cli.ftl`); matched at the start of the result.
FAILURE_PREFIXES = ("Error: ", "Error executing ", "Unknown tool: ",
                    "Tool not available in this turn: ", "Denied by user.", "Cancelled by hook: ",
                    "Skipped duplicate tool call ",
                    "[interrupted by user before this tool produced a result]",
                    "Skipped an additional identical call in this recovery batch.")
INFO_FIELDS = ("session_uuid", "agent_alias", "workspace_dir", "interaction_surface",
               "created_at", "last_activity", "killed_at", "token_count")
_TOOL_NAMES = {"shell": "Bash", "file_read": "Read", "file_write": "Write", "file_edit": "Edit",
               "glob_search": "Glob", "content_search": "Grep"}


def zeroclaw_config_dir(env=None):
    """The ZeroClaw install the environment names, as ZeroClaw resolves it; None without a home."""
    env = os.environ if env is None else env
    custom = str(env.get("ZEROCLAW_CONFIG_DIR") or "").strip()
    if custom:
        return os.path.expanduser(custom)
    data = str(env.get("ZEROCLAW_DATA_DIR") or "").strip() or str(
        env.get("ZEROCLAW_WORKSPACE") or "").strip()
    if data:
        data = os.path.expanduser(data)
        if os.path.isfile(os.path.join(data, "config.toml")):
            return data
        legacy = os.path.join(os.path.dirname(data.rstrip(os.sep)), ".zeroclaw")
        if os.path.isfile(os.path.join(legacy, "config.toml")) or os.path.basename(
                data.rstrip(os.sep)) in ("data", "workspace"):
            return legacy
        return data
    home = env.get("HOME") if os.name != "nt" else env.get("USERPROFILE")
    return os.path.join(home, ".zeroclaw") if home else None


def zeroclaw_db(config_dir):
    return os.path.join(config_dir, *DB_PARTS) if config_dir else None


def zeroclaw_epoch(ts):
    """Epoch seconds of one of ZeroClaw's RFC 3339 times; None if it is not one."""
    if not isinstance(ts, str) or not ts:
        return None
    text = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    date, dot, rest = text.partition(".")
    if dot:
        # Python before 3.11 reads at most six digits of a fraction; ZeroClaw writes nine.
        digits = len(rest) - len(rest.lstrip("0123456789"))
        text = f"{date}.{rest[:min(digits, 6)]}{rest[digits:]}"
    try:
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def zeroclaw_tool(name, args):
    """(name, args) of a call as the mechanisms read it; any other tool as it is."""
    args = args if isinstance(args, dict) else {}
    mapped = _TOOL_NAMES.get(name or "")
    if mapped == "Bash":
        return mapped, {"command": str(args.get("command") or "")}
    if mapped in ("Read", "Write", "Edit"):
        return mapped, {"file_path": str(args.get("path") or args.get("file_path") or "")}
    if mapped in ("Glob", "Grep"):
        return mapped, {"pattern": str(args.get("pattern") or args.get("query") or ""),
                        "path": str(args.get("path") or "")}
    return name or "", args


def zeroclaw_args(payload):
    """An `in` row's arguments (a JSON object in a string), {} when they are not one."""
    if isinstance(payload, dict):
        return payload
    try:
        args = json.loads(payload or "{}")
    except (TypeError, ValueError):
        return {}
    return args if isinstance(args, dict) else {}


def zeroclaw_result(text):
    """(text, failed) of one tool result as the model read it."""
    text = text if isinstance(text, str) else str(text or "")
    return text, text.lstrip().startswith(FAILURE_PREFIXES)


def strip_date_header(text):
    """A prompt without the `[CURRENT DATE & TIME: ...]` line ZeroClaw puts before every one
    (`crates/zeroclaw-runtime/src/agent/agent.rs`)."""
    if isinstance(text, str) and text.startswith("[CURRENT DATE & TIME:"):
        _, sep, rest = text.partition("]\n\n")
        return rest if sep else ""
    return text


def _load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, TypeError):
        return {}, []
    if not isinstance(data, dict):
        return {}, []
    msgs = data.get("messages")
    msgs = [m for m in msgs if isinstance(m, dict)] if isinstance(msgs, list) else []
    return (data.get("info") if isinstance(data.get("info"), dict) else {}), msgs


def _is_prompt(m):
    return (m.get("role") == "user" and str(m.get("content") or "").strip() != ""
            and m.get("content") != TRIM_BREADCRUMB)


def zeroclaw_session_info(path):
    return _load(path)[0]


def zeroclaw_prompt_count(path):
    """How many prompts the conversation ran."""
    return sum(1 for m in _load(path)[1] if _is_prompt(m))


def zeroclaw_session_to_claude(path, out_path):
    """Write an export as Claude Code transcript lines. Returns the number of tool calls written,
    or None when the file is missing or holds no messages."""
    if not path or not os.path.exists(path):
        return None
    info, messages = _load(path)
    if not messages:
        return None
    cwd = info.get("workspace_dir") if isinstance(info.get("workspace_dir"), str) else ""
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for m in messages:
            role, when = m.get("role"), m.get("created_at")
            if role == "user":
                if _is_prompt(m):
                    out.write(json.dumps({"type": "user", "timestamp": when, "message": {
                        "role": "user", "content": str(m.get("content"))}}) + "\n")
                continue
            if role != "assistant":
                continue
            tools = [t for t in m.get("tools") or [] if isinstance(t, dict)]
            calls = [t for t in tools if t.get("event_kind") == "in"]
            results = [t for t in tools if t.get("event_kind") == "out"]
            blocks = []
            for t in calls:
                name, args = zeroclaw_tool(t.get("tool_name"), zeroclaw_args(t.get("payload")))
                blocks.append({"type": "tool_use", "id": t.get("tool_call_id"), "name": name,
                               "input": args})
                n += 1
            if blocks:
                out.write(json.dumps(dict({"type": "assistant", "timestamp": when, "message": {
                    "role": "assistant", "content": blocks, "model": None}},
                    **({"cwd": cwd} if cwd else {}))) + "\n")
            for t in results:
                text, failed = zeroclaw_result(t.get("payload"))
                out.write(json.dumps({"type": "user", "timestamp": when, "message": {
                    "role": "user", "content": [{"type": "tool_result",
                                                 "tool_use_id": t.get("tool_call_id"),
                                                 "content": text, "is_error": failed}]}}) + "\n")
    return n


def zeroclaw_session_usage(path):
    """ZeroClaw stores no token usage per session: the unpriced shape, with its request count."""
    requests = sum(1 for m in _load(path)[1] if m.get("role") == "assistant")
    return {"input_tokens": None, "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None, "output_tokens": None, "requests": requests,
            "model": None}


# -- the live database ---------------------------------------------------------------------
def _connect(db_path):
    con = sqlite3.connect(pathlib.Path(db_path).absolute().as_uri() + "?mode=ro", uri=True,
                          timeout=5)
    con.row_factory = sqlite3.Row
    return con


def _private(row):
    """A message as the export holds it: a prompt counted, never kept; the model's words dropped."""
    role, content = row["role"], row["content"]
    if role == "user":
        content = (content if content == TRIM_BREADCRUMB else
                   PROMPT if str(strip_date_header(content) or "").strip() else "")
    else:
        content = ""
    return {"role": role, "content": content, "created_at": row["created_at"], "tools": []}


def zeroclaw_export(db_path, session_uuid, path):
    """Write a session's rows as the JSON this module reads; whether it was written."""
    try:
        con = _connect(db_path)
        try:
            row = con.execute("SELECT * FROM acp_sessions WHERE session_uuid = ?",
                              (session_uuid,)).fetchone()
            if not row:
                return False
            msgs = con.execute("SELECT id, role, content, created_at FROM acp_messages WHERE "
                               "session_id = ? ORDER BY id", (row["id"],)).fetchall()
            tools = con.execute(
                "SELECT tc.message_id, tc.tool_call_id, tc.tool_name, tc.event_kind, tc.payload "
                "FROM acp_tool_calls tc JOIN acp_messages m ON m.id = tc.message_id "
                "WHERE m.session_id = ? ORDER BY tc.id", (row["id"],)).fetchall()
        finally:
            con.close()
        info = {k: row[k] for k in INFO_FIELDS if k in row.keys()}
        by_id, messages = {}, []
        for m in msgs:
            if m["role"] == INTERRUPTED_ROLE:
                continue
            rec = _private(m)
            by_id[m["id"]] = rec
            messages.append(rec)
        for t in tools:
            rec = by_id.get(t["message_id"])
            if rec is not None:
                rec["tools"].append({k: t[k] for k in ("tool_call_id", "tool_name", "event_kind",
                                                       "payload")})
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w",
                  encoding="utf-8") as fh:
            json.dump({"info": info, "messages": messages}, fh)
        os.replace(tmp, path)
        return True
    except (OSError, ValueError, TypeError, sqlite3.Error):
        return False
