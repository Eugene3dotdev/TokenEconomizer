"""Factory Droid's session log, read as history only (the plugin runs no hooks in Droid).

Droid keeps one file per session, `~/.factory/sessions/<folder>/<session id>.jsonl`, where
`<folder>` is the working directory with every character that is not a letter or a digit turned
into `-` (`/home/u/widget` -> `-home-u-widget`, the same rule as Claude Code's). Verified on
Droid 1.5.0 logs written 2026-11-03 to 2026-11-05:

  * the first line is `{"type": "session_start", "id", "cwd", "title", "owner", "version"}`;
  * a conversation entry is `{"type": "message", "id", "parentId", "timestamp", "message":
    {"role", "content", ...}}`, whose `content` is the Anthropic block list Claude Code also
    writes: `text`, `thinking`, `tool_use` (`id`, `name`, `input`) and `tool_result`
    (`tool_use_id`, `content`, optional `is_error`). An assistant entry names its `modelId`;
    no entry carries token usage, so a Droid session is unpriced;
  * `todo_state`, `agent_turn_outcome`, `compaction_state` and the like are Droid's own
    bookkeeping and are not conversation;
  * Droid injects context as `user` entries of its own: their `id` starts with `context-`
    (tool catalog updates, system reminders) or `repl-tool-result:` (a result of a tool the
    model ran inside its `Script` tool). Neither is a prompt the person typed.

`droid_session_to_claude` rewraps each conversation entry as a Claude Code transcript line, so
every reader built for Claude Code's log handles a Droid session. Droid's injected entries are
written with `isMeta`, which keeps them out of the prompt count and the prompt-turns.
"""
import json
import os
import re
from datetime import datetime

from .jsonl import lines as _lines

INJECTED_ID_PREFIXES = ("context-", "repl-tool-result:")
_TAIL_BYTES = 64 * 1024


def droid_sessions_dir(env=None):
    """Where Droid keeps its session logs: `$FACTORY_HOME/sessions`, else `~/.factory/sessions`."""
    env = os.environ if env is None else env
    base = env.get("FACTORY_HOME")
    if not base:
        home = env.get("HOME") if os.name != "nt" else env.get("USERPROFILE")
        base = os.path.join(home or os.path.expanduser("~"), ".factory")
    return os.path.join(base, "sessions")


def droid_history_logs(sessions_dir):
    """[(folder, path, mtime, size)] for every session log under `sessions_dir`."""
    out = []
    try:
        folders = sorted(os.listdir(sessions_dir))
    except OSError:
        return out
    for folder in folders:
        d = os.path.join(sessions_dir, folder)
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for name in names:
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(d, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if os.path.isfile(path):
                out.append((folder, path, st.st_mtime, st.st_size))
    return out


def _epoch(ts):
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts[:-1] + "+00:00" if ts.endswith("Z") else ts).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _injected(ev):
    return str(ev.get("id") or "").startswith(INJECTED_ID_PREFIXES)


def _has_result(content):
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _is_prompt(ev):
    """A `user` message the person gave: not injected by Droid, not a tool result."""
    if ev.get("type") != "message" or _injected(ev):
        return False
    msg = ev.get("message") or {}
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    blocks = [b for b in content if isinstance(b, dict)]
    if any(b.get("type") == "tool_result" for b in blocks):
        return False
    return any(b.get("type") == "text" and str(b.get("text") or "").strip() for b in blocks)


def droid_log_head(path):
    """{"session_id", "cwd", "started", "ended"} from a log's first lines and its tail; each
    None where the log does not say. `started` is the first message's time, `ended` the last."""
    out = {"session_id": None, "cwd": None, "started": None, "ended": None}
    first = None
    for line in _lines(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "session_start":
            out["session_id"] = ev.get("id") if isinstance(ev.get("id"), str) else None
            out["cwd"] = ev.get("cwd") if isinstance(ev.get("cwd"), str) and ev["cwd"] else None
            continue
        if ev.get("type") == "message":
            first = _epoch(ev.get("timestamp"))
            break
    out["started"] = first
    out["ended"] = _last_timestamp(path) or first
    return out


def _last_timestamp(path):
    """The newest `timestamp` in the log's last 64 KB."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - _TAIL_BYTES))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    found = re.findall(r'"timestamp":\s*"([^"]+)"', tail)
    for ts in reversed(found):
        t = _epoch(ts)
        if t is not None:
            return t
    return None


def droid_prompt_count(path):
    """How many prompts the person gave a Droid session (`_is_prompt`)."""
    n = 0
    for line in _lines(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        n += isinstance(ev, dict) and _is_prompt(ev)
    return n


def droid_session_to_claude(path, out_path):
    """Write a Droid session as Claude Code transcript lines; the number of tool calls written,
    or None when the file holds no conversation."""
    head = droid_log_head(path)
    cwd = head.get("cwd")
    n, out = 0, []
    for line in _lines(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "message":
            continue
        msg = ev.get("message") or {}
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            n += sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
        entry = {"type": role, "message": {"role": role, "content": content},
                 "timestamp": ev.get("timestamp"), "sessionId": head.get("session_id")}
        if cwd:
            entry["cwd"] = cwd
        if role == "assistant":
            entry["message"]["id"] = ev.get("id")
            if msg.get("modelId"):
                entry["message"]["model"] = msg["modelId"]
        elif not _is_prompt(ev) and not _has_result(content):
            # Injected context, or an entry with no text (Droid writes some with an empty
            # content list): not a prompt, so not a turn of its own.
            entry["isMeta"] = True
        out.append(entry)
    if not out:
        return None
    with open(out_path, "w", encoding="utf-8") as fh:
        for item in out:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return n


def droid_session_usage(path):
    """Droid's log records no token usage: unpriced, with the model it named."""
    model = None
    for line in _lines(path):
        if '"modelId"' not in line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        m = (ev.get("message") or {}) if isinstance(ev, dict) else {}
        if isinstance(m.get("modelId"), str):
            model = m["modelId"]
            break
    return {"input_tokens": None, "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None, "output_tokens": None,
            "requests": 0, "model": model}
