"""Muse Code's session log, for the live fold of a Muse session and for history.

Muse Code (`muse`, Meta) keeps one directory per session under its data root,
`${XDG_DATA_HOME:-~/.local/share}/muse/sessions/YYYY/MM/DD/<session id>/session.jsonl`, the
date being the session's start (its id is a UUIDv7, whose first 48 bits are that time in
milliseconds). Verified on 1.4.20 (2026-11-16) in a disposable profile:

  * each line is a record `{schema_version, id, stream: {kind, id}, sequence, recorded_at
    (epoch microseconds), record_type, payload_type, payload}`. The first line can instead be
    a `retained_frame` holding records as JSON strings (`children[].record_json`), and a
    `retained_marker` line stands for a record Muse kept only live; neither is conversation;
  * `runtime.session.metadata` names the session's `workspace_root`, `model_id` and
    `provider_id` (`payload.record`);
  * `runtime.session` records carry the run's events in `payload.event`, by `kind`: `started`
    with the person's `prompt` (a run's other `started` events name only a `task_id`),
    `assistant_message_committed` (`text`), `assistant_tool_calls_committed` (`tool_calls`:
    `{name, call_id, args}`, `args` a JSON string), `tool_result_batch_committed` (`results`:
    `{tool_call_id, text}`; a shell result's text is a JSON object with `exit_code`,
    `terminal_status` and `output`), and `model_completed` (`model`, `usage`);
  * a `model_completed` usage counts `input_tokens` with the cached part in it, and
    `cache_read_tokens` / `cache_write_tokens` beside it. Only the main run's model calls are
    there: the reminder agents Muse runs beside a turn (`goal_usage_attribution` records with
    an `owner_type` of `reminder_child`) are not counted.

`muse_session_to_claude` writes a session as Claude Code transcript lines, with Muse's tool
names in Claude Code's vocabulary (`muse_tool`), so every reader built for Claude Code's log
handles a Muse session.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone

from .calls import strip_rtk_prefix
from .jsonl import records as _records

_NAMES = {"bash": "Bash", "read_file": "Read", "write_file": "Write", "edit_file": "Edit",
          "apply_patch": "apply_patch", "search": "Grep", "web_fetch": "WebFetch",
          "web_search": "WebSearch"}
_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File:\s*(.+?)\s*$", re.M)
_HEAD_RECORDS = 64


def muse_data_dir(env=None):
    """Muse's data root: the one around `$MUSE_PLUGIN_DATA_DIR` (`<root>/plugins/data/<plugin>`)
    in a plugin hook, else `$XDG_DATA_HOME/muse`, else `~/.local/share/muse`. Muse 1.4.20 runs
    hooks with a filtered environment that keeps `HOME` and its own `MUSE_*` but drops
    `XDG_DATA_HOME`, so in a hook only the plugin's data directory says which root it runs on."""
    env = os.environ if env is None else env
    plugin_data = env.get("MUSE_PLUGIN_DATA_DIR")
    if plugin_data:
        parts = os.path.normpath(plugin_data).split(os.sep)
        if len(parts) > 3 and parts[-3:-1] == ["plugins", "data"]:
            return os.sep.join(parts[:-3])
    base = env.get("XDG_DATA_HOME")
    if not base:
        home = env.get("HOME") if os.name != "nt" else env.get("USERPROFILE")
        base = os.path.join(home or os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "muse")


def muse_sessions_dir(env=None):
    return os.path.join(muse_data_dir(env), "sessions")


def muse_tool(name, args):
    """(name, args) of a Muse tool call as the mechanisms read it; any other tool as it is."""
    args = args if isinstance(args, dict) else {}
    mapped = _NAMES.get(name or "")
    if mapped == "Bash":
        return mapped, {"command": strip_rtk_prefix(str(args.get("command") or ""))}
    if mapped in ("Read", "Write", "Edit"):
        return mapped, {"file_path": str(args.get("path") or args.get("file_path") or "")}
    if mapped == "Grep":
        return mapped, {"pattern": str(args.get("pattern") or args.get("query") or ""),
                        "path": str(args.get("path") or "")}
    if mapped == "apply_patch":
        paths = _PATCH_FILE_RE.findall(str(args.get("input") or args.get("patch") or ""))
        return mapped, {"path": paths[0] if paths else "", "paths": paths}
    return mapped or name or "", args


def muse_result(text):
    """(text, exit_code_or_None, failed) of a tool result. A shell result is a JSON object
    (`output`, `exit_code`, `terminal_status`); anything else is taken as it is, not failed."""
    if isinstance(text, dict):
        doc = text
    else:
        text = "" if text is None else str(text)
        try:
            doc = json.loads(text) if text.lstrip().startswith("{") else None
        except ValueError:
            doc = None
    if not isinstance(doc, dict):
        return text, None, False
    code = doc.get("exit_code")
    code = code if isinstance(code, int) and not isinstance(code, bool) else None
    out = doc.get("output")
    out = out if isinstance(out, str) else (text if isinstance(text, str) else json.dumps(doc))
    status = str(doc.get("terminal_status") or "")
    failed = (code is not None and code != 0) or status in ("failed", "error", "cancelled",
                                                            "timed_out")
    return out, code, failed


def _uuid7_day(session_id):
    """The UTC date a UUIDv7 session id was made, or None."""
    hexpart = str(session_id or "").replace("-", "")
    if len(hexpart) != 32 or not re.fullmatch(r"[0-9a-fA-F]+", hexpart):
        return None
    try:
        return datetime.fromtimestamp(int(hexpart[:12], 16) / 1000.0, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def muse_find_session(sessions_dir, session_id):
    """The `session.jsonl` of a session, or None. The day its id was made is tried first (in
    UTC and the day either side, for a folder dated in local time), then every day."""
    if not sessions_dir or not re.fullmatch(r"[\w.-]+", str(session_id or "")):
        return None
    t = _uuid7_day(session_id)
    if t is not None:
        for delta in (0, -1, 1):
            d = t + timedelta(days=delta)
            path = os.path.join(sessions_dir, f"{d.year:04d}", f"{d.month:02d}",
                                f"{d.day:02d}", session_id, "session.jsonl")
            if os.path.isfile(path):
                return path
    for _sid, path, _m, _s in muse_history_logs(sessions_dir, only=session_id):
        return path
    return None


def _listdir(d):
    try:
        return sorted(os.listdir(d))
    except OSError:
        return []


def muse_history_logs(sessions_dir, only=None):
    """[(session id, path, mtime, size)] for every session log under `sessions_dir`."""
    out = []
    for y in _listdir(sessions_dir):
        for m in _listdir(os.path.join(sessions_dir, y)):
            for d in _listdir(os.path.join(sessions_dir, y, m)):
                day = os.path.join(sessions_dir, y, m, d)
                for sid in ([only] if only else _listdir(day)):
                    path = os.path.join(day, sid, "session.jsonl")
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    out.append((sid, path, st.st_mtime, st.st_size))
    return out


def _unwrapped(path):
    """Every record of the log in order, those inside a `retained_frame` included."""
    for rec in _records(path):
        if rec.get("retained_frame"):
            for child in rec.get("children") or []:
                raw = child.get("record_json") if isinstance(child, dict) else None
                try:
                    inner = json.loads(raw) if isinstance(raw, str) else None
                except ValueError:
                    inner = None
                if isinstance(inner, dict):
                    yield inner
            continue
        if "payload_type" in rec:
            yield rec


def _event(rec):
    if rec.get("payload_type") != "runtime.session":
        return None
    payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
    ev = payload.get("event")
    return ev if isinstance(ev, dict) else None


def _prompt(ev):
    p = ev.get("prompt") if ev.get("kind") == "started" else None
    return p if isinstance(p, str) and p.strip() else None


def _epoch(rec):
    t = rec.get("recorded_at")
    return t / 1e6 if isinstance(t, (int, float)) and not isinstance(t, bool) else None


def muse_session_head(path):
    """{"session_id", "cwd", "model", "started"} from a log's first records; each None where
    they do not say."""
    out = {"session_id": None, "cwd": None, "model": None, "started": None}
    for i, rec in enumerate(_unwrapped(path)):
        if i >= _HEAD_RECORDS:
            break
        stream = rec.get("stream") if isinstance(rec.get("stream"), dict) else {}
        if out["session_id"] is None and stream.get("kind") == "session" and \
                isinstance(stream.get("id"), str):
            out["session_id"] = stream["id"]
        if out["started"] is None:
            out["started"] = _epoch(rec)
        if rec.get("payload_type") == "runtime.session.metadata":
            meta = (rec.get("payload") or {}).get("record") or {}
            if isinstance(meta, dict):
                root = meta.get("workspace_root")
                out["cwd"] = root if isinstance(root, str) and root else None
                out["model"] = meta.get("model_id") if isinstance(meta.get("model_id"),
                                                                  str) else None
            break
    return out


def muse_prompt_count(path):
    """How many prompts the person gave the session: runs started with a prompt."""
    return sum(1 for rec in _unwrapped(path) if (_event(rec) or {}).get("kind") == "started"
               and _prompt(_event(rec)))


def _args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        doc = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return {}
    return doc if isinstance(doc, dict) else {}


def muse_session_to_claude(path, out_path):
    """Write a Muse session as Claude Code transcript lines; the number of tool calls written,
    or None when the log holds no prompt and no call. A shell call is placed in its `workdir`
    where it names one, else in the session's workspace."""
    head = muse_session_head(path)
    cwd, sid, model = head.get("cwd"), head.get("session_id"), head.get("model")
    n, out = 0, []
    for rec in _unwrapped(path):
        ev = _event(rec)
        if not ev:
            continue
        kind = ev.get("kind")
        base = {"timestamp": rec.get("recorded_at"), "sessionId": sid}
        if kind == "started" and _prompt(ev):
            out.append(dict(base, type="user", message={"role": "user", "content": _prompt(ev)},
                            **({"cwd": cwd} if cwd else {})))
        elif kind == "assistant_message_committed" and isinstance(ev.get("text"), str):
            msg = {"id": str(ev.get("message_id") or ""), "role": "assistant",
                   "content": [{"type": "text", "text": ev["text"]}]}
            if model:
                msg["model"] = model
            out.append(dict(base, type="assistant", message=msg,
                            **({"cwd": cwd} if cwd else {})))
        elif kind == "assistant_tool_calls_committed":
            blocks, where = [], cwd
            for c in ev.get("tool_calls") or []:
                if not isinstance(c, dict):
                    continue
                raw = _args(c.get("args"))
                name, args = muse_tool(c.get("name"), raw)
                if name == "Bash" and isinstance(raw.get("workdir"), str) and \
                        os.path.isabs(raw["workdir"]):
                    where = raw["workdir"]
                blocks.append({"type": "tool_use", "id": c.get("call_id") or c.get("id"),
                               "name": name, "input": args})
                n += 1
            if blocks:
                msg = {"id": str(ev.get("message_id") or ""), "role": "assistant",
                       "content": blocks}
                if model:
                    msg["model"] = model
                out.append(dict(base, type="assistant", message=msg,
                                **({"cwd": where} if where else {})))
        elif kind == "tool_result_batch_committed":
            for r in ev.get("results") or []:
                if not isinstance(r, dict):
                    continue
                text, _code, failed = muse_result(r.get("text"))
                out.append(dict(base, type="user", message={"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r.get("tool_call_id"),
                     "content": text, "is_error": failed}]}))
    if not out:
        return None
    with open(out_path, "w", encoding="utf-8") as fh:
        for item in out:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return n


def muse_session_usage(path):
    """Token breakdown of the session's own model calls (`model_completed`), with the cached
    part taken out of `input_tokens`; the model is the one that answered most of them."""
    totals = {"input_tokens": 0, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "output_tokens": 0}
    requests, models = 0, {}
    for rec in _unwrapped(path):
        ev = _event(rec)
        if not ev or ev.get("kind") != "model_completed" or not isinstance(ev.get("usage"), dict):
            continue
        u = ev["usage"]

        def num(*keys):
            for k in keys:
                v = u.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return int(v)
            return 0
        cached = num("cache_read_tokens", "cached_tokens")
        totals["input_tokens"] += max(0, num("input_tokens") - cached)
        totals["cache_read_input_tokens"] += cached
        totals["cache_creation_input_tokens"] += num("cache_write_tokens")
        totals["output_tokens"] += num("output_tokens")
        requests += 1
        if isinstance(ev.get("model"), str):
            models[ev["model"]] = models.get(ev["model"], 0) + 1
    if not requests:
        return {"input_tokens": None, "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None, "output_tokens": None, "requests": 0,
                "model": muse_session_head(path).get("model")}
    totals.update(requests=requests, model=max(models, key=models.get) if models else None)
    return totals
