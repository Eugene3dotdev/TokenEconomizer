"""Every agent's own session store, behind one read-only contract, for importing history.

A `Source` finds the session logs an agent keeps on this machine (`logs`), says what a log is
about from its head alone (`head`: the directory it ran in, its id, when it started), and turns
one into a trace every mechanism reads (`to_trace`, a Claude-shaped JSONL or Codex's exec
stream). Nothing here knows repositories: a source hands back the `cwd` a log records and the
plugin keys it. Nothing here imports the plugin.

Invariants every source keeps:

  * `logs` touches directory listings and `stat` only, never a log's content; a store that is
    not there yields nothing;
  * `head` reads a log's first lines (or one database row), never the body;
  * `to_trace` writes only under the `workdir` it is given, and returns None for a log it
    cannot read, so that the caller counts it as unreadable rather than half-folds it;
  * a session's end is its file's mtime (a row's `ended_at` for a database), the same clock
    the plugin's own hooks record a live session's end with.

`headless` marks a one-shot scripted run (`claude -p`, `codex exec`, `hermes -z`): with one
prompt it says little about the person's own sessions, and the plan leaves it out.

Sources are registered by name in `SOURCES`; a `Host` of the same name is the agent the plugin
runs hooks in. An agent can have a source and no host (Factory Droid): its sessions teach every
other agent's store for the repository, and it has no store of its own.
"""
import json
import os
import re
import sqlite3
from urllib.parse import unquote

from . import claude as _claude
from . import codex as _codex
from . import copilot as _copilot
from . import droid as _droid
from . import grok as _grok
from . import hermes as _hermes
from . import opencode as _opencode
from . import pi as _pi
from .jsonl import lines as _lines

SOURCES = {}


def register(source):
    SOURCES[source.name] = source
    return source


def _home(env):
    """The home the environment names, or None: a store is found only through the environment
    a source is given, so a hook that is handed a bare one reads nothing of this machine."""
    env = os.environ if env is None else env
    return env.get("HOME") if os.name != "nt" else env.get("USERPROFILE")


def _base(env, key, *parts):
    """`$key` when the environment sets it, else `~/<parts>`; None with neither."""
    env = os.environ if env is None else env
    if key and env.get(key):
        return env[key]
    home = _home(env)
    return os.path.join(home, *parts) if home else None


def _stat_logs(paths, ids):
    out = []
    for path, sid in zip(paths, ids):
        try:
            st = os.stat(path)
        except OSError:
            continue
        if os.path.isfile(path):
            out.append({"id": sid, "path": path, "mtime": st.st_mtime, "size": st.st_size})
    return out


def _first_record(path):
    for line in _lines(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            return ev
        return None
    return None


def _epoch(ts):
    return _claude._epoch(ts)


class Source:
    """One agent's session store on this machine; see the module docstring for the contract."""
    name = "?"
    label = "?"

    def logs(self, env, extra_dirs=()):
        """[{"id", "path", "mtime", "size"}] for every session log the agent keeps; `id` is
        the agent's own session id (unique within this source), `path` what `head` and
        `to_trace` read (for a database, `<db path>#<row id>`)."""
        return []

    def head(self, path):
        """{"session_id", "cwd", "started", "headless", "folder"}: the directory the session
        ran in as the agent recorded it (None: unknown, so it cannot be placed), its first
        time as epoch seconds (None where the head does not say), whether it was a one-shot
        scripted run, and, where the agent files logs by directory, that folder's name."""
        return {"session_id": None, "cwd": None, "started": None, "headless": False,
                "folder": None}

    def prompt_count(self, path):
        """How many prompts the person gave; None where only the body knows."""
        return None

    def materialize(self, path, workdir):
        """A file `to_trace` reads: the log itself, or for a database one export written
        under `workdir`. None when it cannot be written now (the plugin tries later)."""
        return path

    def to_trace(self, path, workdir):
        """Write the trace `trace.load_trajectory` reads under `workdir`; its path, or None."""
        return None

    def usage(self, path):
        """Token usage for the ledger; the unpriced shape where the agent records none."""
        return {"input_tokens": None, "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None, "output_tokens": None, "requests": 0,
                "model": None}

    turns = None       # `turns(path)`: prompt-turns in `claude_prompt_turns` shape, or absent


# ---- Claude Code -----------------------------------------------------------------------------
class ClaudeSource(Source):
    name, label = "claude", "Claude Code"

    def logs(self, env, extra_dirs=()):
        env = os.environ if env is None else env
        if not env.get("CLAUDE_CONFIG_DIR") and not _home(env):
            return []
        dirs = [_claude.claude_projects_dir(env)]
        for d in extra_dirs or ():
            d = os.path.expanduser(str(d))
            dirs.append(os.path.join(d, "projects") if os.path.isdir(os.path.join(d, "projects"))
                        else d)
        out, seen, seen_dirs = [], set(), set()
        for d in dirs:
            real = os.path.realpath(d)
            if real in seen_dirs:
                continue
            seen_dirs.add(real)
            for folder, path, mtime, size in _claude.claude_history_logs(d):
                sid = os.path.basename(path)[:-len(".jsonl")]
                if sid in seen:
                    continue
                seen.add(sid)
                out.append({"id": sid, "path": path, "mtime": mtime, "size": size,
                            "folder": folder})
        return out

    def head(self, path):
        h = _claude.claude_log_head(path)
        return {"session_id": h.get("session_id"), "cwd": h.get("cwd"),
                "started": h.get("started"),
                "headless": str(h.get("entrypoint") or "").lower().startswith("sdk"),
                "folder": os.path.basename(os.path.dirname(path))}

    def prompt_count(self, path):
        return _claude.claude_prompt_count(path)

    def to_trace(self, path, workdir):
        return path if os.path.exists(path) else None

    def usage(self, path):
        return _claude.claude_session_usage(path)

    def turns(self, path):
        return _claude.claude_prompt_turns(path)


# ---- Codex -----------------------------------------------------------------------------------
_ROLLOUT_RE = re.compile(r"^rollout-.*-([0-9a-fA-F-]{36})\.jsonl$")


def codex_session_roots(env):
    """Every `sessions` root Codex writes rollouts under: `$CODEX_HOME/sessions` (else
    `~/.codex/sessions`), and each OpenClaw agent's embedded Codex home."""
    env = os.environ if env is None else env
    codex_home = _base(env, "CODEX_HOME", ".codex")
    roots = [os.path.join(codex_home, "sessions")] if codex_home else []
    home = _home(env)
    agents = os.path.join(home, ".openclaw", "agents") if home else None
    try:
        names = sorted(os.listdir(agents)) if agents else []
    except OSError:
        names = []
    for n in names:
        roots.append(os.path.join(agents, n, "agent", "codex-home", "sessions"))
    return roots


class CodexSource(Source):
    name, label = "codex", "Codex"

    def logs(self, env, extra_dirs=()):
        roots = codex_session_roots(env) + [os.path.expanduser(str(d)) for d in extra_dirs or ()
                                            if os.path.isdir(os.path.join(
                                                os.path.expanduser(str(d)), "sessions"))
                                            or "codex" in str(d)]
        paths, ids, seen = [], [], set()
        for root in roots:
            real = os.path.realpath(root)
            if real in seen or not os.path.isdir(root):
                continue
            seen.add(real)
            for dirpath, _dirs, names in os.walk(root):
                for n in sorted(names):
                    m = _ROLLOUT_RE.match(n)
                    if m:
                        paths.append(os.path.join(dirpath, n))
                        ids.append(m.group(1))
        return _stat_logs(paths, ids)

    def head(self, path):
        ev = _first_record(path) or {}
        p = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        meta = p if ev.get("type") == "session_meta" else {}
        sid = meta.get("id") or meta.get("session_id")
        return {"session_id": sid if isinstance(sid, str) else None,
                "cwd": meta.get("cwd") if isinstance(meta.get("cwd"), str) else None,
                "started": _epoch(meta.get("timestamp") or ev.get("timestamp")),
                "headless": (meta.get("source") == "exec"
                             or str(meta.get("originator") or "") == "codex_exec"),
                "folder": None}

    def prompt_count(self, path):
        return _codex.codex_prompt_count(path)

    def to_trace(self, path, workdir):
        out = os.path.join(workdir, "codex_exec_trace.jsonl")
        n = _codex.codex_rollout_to_exec(path, out)
        return out if n is not None else None

    def usage(self, path):
        return _codex.codex_session_usage(path)


# ---- GitHub Copilot CLI ----------------------------------------------------------------------
class CopilotSource(Source):
    name, label = "copilot", "Copilot CLI"

    def logs(self, env, extra_dirs=()):
        base = _base(env, "COPILOT_HOME", ".copilot")
        if not base:
            return []
        root = os.path.join(base, "session-state")
        try:
            names = sorted(os.listdir(root))
        except OSError:
            return []
        paths = [os.path.join(root, n, "events.jsonl") for n in names]
        return _stat_logs(paths, names)

    def head(self, path):
        meta = _copilot.copilot_session_meta(path)
        ev = _first_record(path) or {}
        started = _epoch(((ev.get("data") or {}).get("startTime")) or ev.get("timestamp"))
        return {"session_id": meta.get("session_id"), "cwd": meta.get("cwd"),
                "started": started, "headless": False, "folder": None}

    def prompt_count(self, path):
        return _copilot.copilot_prompt_count(path)

    def to_trace(self, path, workdir):
        return path if os.path.exists(path) else None

    def usage(self, path):
        return _copilot.copilot_session_usage(path)


# ---- Pi ----------------------------------------------------------------------------------------
_PI_FILE_RE = re.compile(r"^(.+?)_([0-9a-fA-F-]{36})\.jsonl$")


class PiSource(Source):
    name, label = "pi", "Pi"

    def logs(self, env, extra_dirs=()):
        base = _base(env, "PI_CODING_AGENT_DIR", ".pi", "agent")
        if not base:
            return []
        root = os.path.join(base, "sessions")
        paths, ids = [], []
        try:
            folders = sorted(os.listdir(root))
        except OSError:
            return []
        for folder in folders:
            d = os.path.join(root, folder)
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for n in names:
                m = _PI_FILE_RE.match(n)
                if m:
                    paths.append(os.path.join(d, n))
                    ids.append(m.group(2))
        return _stat_logs(paths, ids)

    def head(self, path):
        h = _pi.pi_session_header(path)
        return {"session_id": h.get("id") if isinstance(h.get("id"), str) else None,
                "cwd": h.get("cwd") if isinstance(h.get("cwd"), str) else None,
                "started": _epoch(h.get("timestamp")), "headless": False, "folder": None}

    def prompt_count(self, path):
        return _pi.pi_prompt_count(path)

    def to_trace(self, path, workdir):
        out = os.path.join(workdir, "pi_trace.jsonl")
        n = _pi.pi_session_to_claude(path, out)
        return out if n is not None else None

    def usage(self, path):
        return _pi.pi_session_usage(path)


# ---- Grok Build ------------------------------------------------------------------------------
class GrokSource(Source):
    """`~/.grok/sessions/<url-encoded cwd>/<session id>/updates.jsonl`: the ACP update stream
    `trace.grok` reads; the directory name is the session's cwd."""
    name, label = "grok", "Grok Build"

    def logs(self, env, extra_dirs=()):
        base = _base(env, "GROK_HOME", ".grok")
        if not base:
            return []
        root = os.path.join(base, "sessions")
        paths, ids = [], []
        try:
            folders = sorted(os.listdir(root))
        except OSError:
            return []
        for folder in folders:
            d = os.path.join(root, folder)
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for sid in names:
                paths.append(os.path.join(d, sid, "updates.jsonl"))
                ids.append(sid)
        return _stat_logs(paths, ids)

    def head(self, path):
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        sid = os.path.basename(os.path.dirname(path))
        ev = _first_record(path) or {}
        ts = ev.get("timestamp")
        started = float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else None
        cwd = unquote(folder) if folder else ""
        return {"session_id": sid or None, "cwd": cwd if os.path.isabs(cwd) else None,
                "started": started, "headless": False, "folder": folder}

    def prompt_count(self, path):
        return _grok.grok_prompt_count(path)

    def to_trace(self, path, workdir):
        out = os.path.join(workdir, "grok_trace.jsonl")
        n = _grok.grok_session_to_claude(path, out)
        return out if n is not None else None

    def usage(self, path):
        return _grok.grok_session_usage(path)


# ---- Hermes Agent ----------------------------------------------------------------------------
def _db_ref(db, rowid):
    return f"{db}#{rowid}"


def _split_ref(path):
    db, _, rowid = str(path).rpartition("#")
    return (db, rowid) if db else (None, None)


def _ro(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


class HermesSource(Source):
    """Rows of `<HERMES_HOME>/state.db`; a session is exported with `hermes_export` (the plugin's
    own copy, prompts replaced) before it is read."""
    name, label = "hermes", "Hermes Agent"

    def _db(self, env):
        base = _base(env, "HERMES_HOME", ".hermes")
        return os.path.join(base, "state.db") if base else None

    def logs(self, env, extra_dirs=()):
        db = self._db(env)
        if not db or not os.path.isfile(db):
            return []
        try:
            con = _ro(db)
            try:
                rows = con.execute(
                    "SELECT id, started_at, ended_at, message_count FROM sessions WHERE "
                    "source IS NOT 'subagent' AND parent_session_id IS NULL "
                    "ORDER BY started_at").fetchall()
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            return []
        out = []
        for r in rows:
            sid = r["id"]
            if not isinstance(sid, str):
                continue
            ended = r["ended_at"] or r["started_at"]
            if not isinstance(ended, (int, float)):
                continue
            out.append({"id": sid, "path": _db_ref(db, sid), "mtime": float(ended),
                        "size": int(r["message_count"] or 0) * 1024})
        return out

    def head(self, path):
        db, sid = _split_ref(path)
        if not db:
            return super().head(path)
        try:
            con = _ro(db)
            try:
                row = con.execute("SELECT id, cwd, started_at, source FROM sessions WHERE id = ?",
                                  (sid,)).fetchone()
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            row = None
        if not row:
            return super().head(path)
        return {"session_id": row["id"], "cwd": row["cwd"] if isinstance(row["cwd"], str) and
                row["cwd"] else None, "started": row["started_at"],
                "headless": row["source"] == "oneshot", "folder": None}

    def materialize(self, path, workdir):
        db, sid = _split_ref(path)
        if not db:
            return path if os.path.isfile(str(path)) else None
        out = os.path.join(workdir, f"hermes-{sid}.json")
        return out if _hermes.hermes_export(db, sid, out) else None

    def prompt_count(self, path):
        """From the export, or, for a row reference, counted in the database: a session's
        `user` messages, less the compression summaries Hermes writes as the person's."""
        db, sid = _split_ref(path)
        if not db:
            return _hermes.hermes_prompt_count(path)
        try:
            con = _ro(db)
            try:
                row = con.execute("SELECT count(*) AS n FROM messages WHERE session_id = ? AND "
                                  "role = 'user' AND content IS NOT NULL AND content != ''",
                                  (sid,)).fetchone()
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            return None
        return int(row["n"]) if row else None

    def to_trace(self, path, workdir):
        out = os.path.join(workdir, "hermes_trace.jsonl")
        n = _hermes.hermes_session_to_claude(path, out)
        return out if n is not None else None

    def usage(self, path):
        return _hermes.hermes_session_usage(path)


# ---- opencode --------------------------------------------------------------------------------
class OpenCodeSource(Source):
    """Rows of opencode's `opencode.db`, read only: a session's messages and parts are written
    out as the export `trace.opencode` reads, so no `opencode` process is needed."""
    name, label = "opencode", "opencode"

    def _db(self, env):
        data = _base(env, "XDG_DATA_HOME", ".local", "share")
        return os.path.join(data, "opencode", "opencode.db") if data else None

    def logs(self, env, extra_dirs=()):
        db = self._db(env)
        if not db or not os.path.isfile(db):
            return []
        try:
            con = _ro(db)
            try:
                rows = con.execute("SELECT id, time_created, time_updated FROM session WHERE "
                                   "parent_id IS NULL ORDER BY time_created").fetchall()
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            return []
        out = []
        for r in rows:
            sid = r["id"]
            if not isinstance(sid, str):
                continue
            ended = r["time_updated"] or r["time_created"]
            if not isinstance(ended, (int, float)):
                continue
            out.append({"id": sid, "path": _db_ref(db, sid), "mtime": float(ended) / 1000.0,
                        "size": 0})
        return out

    def head(self, path):
        db, sid = _split_ref(path)
        if not db:
            return super().head(path)
        try:
            con = _ro(db)
            try:
                row = con.execute("SELECT id, directory, time_created FROM session WHERE id = ?",
                                  (sid,)).fetchone()
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            row = None
        if not row:
            return super().head(path)
        created = row["time_created"]
        return {"session_id": row["id"],
                "cwd": row["directory"] if isinstance(row["directory"], str) and row["directory"]
                else None,
                "started": float(created) / 1000.0 if isinstance(created, (int, float)) else None,
                "headless": False, "folder": None}

    def materialize(self, path, workdir):
        db, sid = _split_ref(path)
        if not db:
            return path if os.path.isfile(str(path)) else None
        try:
            con = _ro(db)
            try:
                info = con.execute("SELECT id, directory FROM session WHERE id = ?",
                                   (sid,)).fetchone()
                if not info:
                    return None
                msgs = con.execute("SELECT id, data FROM message WHERE session_id = ? ORDER BY "
                                   "time_created, id", (sid,)).fetchall()
                parts = con.execute("SELECT message_id, data FROM part WHERE session_id = ? "
                                    "ORDER BY time_created, id", (sid,)).fetchall()
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            return None
        by_msg = {}
        for p in parts:
            try:
                d = json.loads(p["data"])
            except (TypeError, ValueError):
                continue
            if isinstance(d, dict):
                by_msg.setdefault(p["message_id"], []).append(d)
        messages = []
        for m in msgs:
            try:
                d = json.loads(m["data"])
            except (TypeError, ValueError):
                continue
            if not isinstance(d, dict):
                continue
            d = dict(d, id=m["id"])
            messages.append({"info": d, "parts": by_msg.get(m["id"], [])})
        out = os.path.join(workdir, f"opencode-{sid}.json")
        try:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump({"info": {"id": info["id"], "directory": info["directory"]},
                           "messages": messages}, fh)
        except OSError:
            return None
        return out

    def prompt_count(self, path):
        """From the export, or, for a row reference, counted in the database: the session's
        `user` messages that carry a text part."""
        db, sid = _split_ref(path)
        if not db:
            return _opencode.opencode_prompt_count(path)
        try:
            con = _ro(db)
            try:
                row = con.execute(
                    "SELECT count(DISTINCT m.id) AS n FROM message m JOIN part p ON "
                    "p.message_id = m.id WHERE m.session_id = ? AND "
                    "json_extract(m.data, '$.role') = 'user' AND "
                    "json_extract(p.data, '$.type') = 'text' AND "
                    "trim(coalesce(json_extract(p.data, '$.text'), '')) != ''", (sid,)).fetchone()
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            return None
        return int(row["n"]) if row else None

    def to_trace(self, path, workdir):
        out = os.path.join(workdir, "opencode_trace.jsonl")
        n = _opencode.opencode_session_to_claude(path, out)
        return out if n is not None else None

    def usage(self, path):
        return _opencode.opencode_session_usage(path)


# ---- Factory Droid (history only; the plugin runs no hooks there) --------------------------
class DroidSource(Source):
    name, label = "droid", "Factory Droid"

    def logs(self, env, extra_dirs=()):
        env = os.environ if env is None else env
        if not env.get("FACTORY_HOME") and not _home(env):
            return []
        out = []
        for folder, path, mtime, size in _droid.droid_history_logs(_droid.droid_sessions_dir(env)):
            out.append({"id": os.path.basename(path)[:-len(".jsonl")], "path": path,
                        "mtime": mtime, "size": size, "folder": folder})
        return out

    def head(self, path):
        h = _droid.droid_log_head(path)
        return {"session_id": h.get("session_id"), "cwd": h.get("cwd"),
                "started": h.get("started"), "headless": False,
                "folder": os.path.basename(os.path.dirname(path))}

    def prompt_count(self, path):
        return _droid.droid_prompt_count(path)

    def to_trace(self, path, workdir):
        out = os.path.join(workdir, "droid_trace.jsonl")
        n = _droid.droid_session_to_claude(path, out)
        return out if n is not None else None

    def usage(self, path):
        return _droid.droid_session_usage(path)


for _s in (ClaudeSource(), CodexSource(), CopilotSource(), PiSource(), GrokSource(),
           HermesSource(), OpenCodeSource(), DroidSource()):
    register(_s)


def enabled(names):
    """The sources a `history_sources` setting names: `all` (every one), `self` (handled by the
    caller: the host's own), or a list of names; unknown names are ignored."""
    words = [str(n).strip().lower() for n in (names or ["all"])]
    if "all" in words:
        return list(SOURCES.values())
    return [SOURCES[n] for n in words if n in SOURCES]
