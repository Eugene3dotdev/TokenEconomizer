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
    out as the export `trace.opencode` reads, so no `opencode` process is needed.

    Two layouts coexist (2.2.x): `session` / `message` / `part`, each message's `data` an
    `{info}` record and its parts rows of their own; and `session_v2` / `session_message` /
    `session_part`, where a message's kind is the `type` column and its `data` is the flat
    export shape (`text` for a prompt; `content` with text and tool entries for an answer). A
    database holds both tables during the move, with the earlier sessions in both; such a
    session is read once, from the newer table."""
    name, label = "opencode", "opencode"
    _SESSION_TABLES = ("session_v2", "session")

    def _db(self, env):
        data = _base(env, "XDG_DATA_HOME", ".local", "share")
        return os.path.join(data, "opencode", "opencode.db") if data else None

    @staticmethod
    def _tables(con):
        return {r["name"] for r in
                con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}

    def _session_row(self, con, sid):
        """(table, row) of the session, from `session_v2` first; (None, None) when absent."""
        tables = self._tables(con)
        for table in self._SESSION_TABLES:
            if table in tables:
                row = con.execute(f"SELECT id, directory, time_created FROM {table} "
                                  "WHERE id = ?", (sid,)).fetchone()
                if row:
                    return table, row
        return None, None

    def logs(self, env, extra_dirs=()):
        db = self._db(env)
        if not db or not os.path.isfile(db):
            return []
        seen, sizes = {}, {}
        try:
            con = _ro(db)
            try:
                tables = self._tables(con)
                for table in self._SESSION_TABLES:
                    if table not in tables:
                        continue
                    rows = con.execute(f"SELECT id, time_created, time_updated FROM {table} "
                                       "WHERE parent_id IS NULL ORDER BY time_created").fetchall()
                    for r in rows:
                        sid, ended = r["id"], r["time_updated"] or r["time_created"]
                        if isinstance(sid, str) and sid not in seen and \
                                isinstance(ended, (int, float)):
                            seen[sid] = float(ended) / 1000.0
                # The bytes a session's rows hold, so the size cap applies to a database
                # session as it does to a log file; one query per table, not per session.
                for table in ("session_message", "message", "part"):
                    if table in tables:
                        for r in con.execute(f"SELECT session_id, sum(length(data)) AS n FROM "
                                             f"{table} GROUP BY session_id").fetchall():
                            if isinstance(r["session_id"], str):
                                sizes[r["session_id"]] = sizes.get(r["session_id"], 0) + \
                                    int(r["n"] or 0)
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            return []
        return [{"id": sid, "path": _db_ref(db, sid), "mtime": mtime, "size": sizes.get(sid, 0)}
                for sid, mtime in sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))]

    def head(self, path):
        db, sid = _split_ref(path)
        if not db:
            return super().head(path)
        try:
            con = _ro(db)
            try:
                _, row = self._session_row(con, sid)
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
                table, info = self._session_row(con, sid)
                if not info:
                    return None
                messages = (self._v2_messages(con, sid, "session_part" in self._tables(con))
                            if table == "session_v2" else self._v1_messages(con, sid))
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            return None
        out = os.path.join(workdir, f"opencode-{sid}.json")
        try:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump({"info": {"id": info["id"], "directory": info["directory"]},
                           "messages": messages}, fh)
        except OSError:
            return None
        return out

    @staticmethod
    def _json_rows(rows, key="data"):
        for r in rows:
            try:
                d = json.loads(r[key])
            except (TypeError, ValueError):
                continue
            if isinstance(d, dict):
                yield r, d

    def _v1_messages(self, con, sid):
        """`message` / `part` rows as `{info, parts}` messages."""
        msgs = con.execute("SELECT id, data FROM message WHERE session_id = ? ORDER BY "
                           "time_created, id", (sid,)).fetchall()
        parts = con.execute("SELECT message_id, data FROM part WHERE session_id = ? "
                            "ORDER BY time_created, id", (sid,)).fetchall()
        by_msg = {}
        for p, d in self._json_rows(parts):
            by_msg.setdefault(p["message_id"], []).append(d)
        return [{"info": dict(d, id=m["id"]), "parts": by_msg.get(m["id"], [])}
                for m, d in self._json_rows(msgs)]

    def _v2_messages(self, con, sid, has_parts):
        """`session_message` rows as the flat export shape (`{id, type, ...}`); parts filed in
        `session_part` (unused by 2.2.x, which embeds them in `content`) are appended there."""
        msgs = con.execute("SELECT id, type, data FROM session_message WHERE session_id = ? "
                           "ORDER BY seq, time_created, id", (sid,)).fetchall()
        by_msg = {}
        if has_parts:
            parts = con.execute("SELECT message_id, data FROM session_part WHERE session_id = ? "
                                "ORDER BY message_id, ordinal", (sid,)).fetchall()
            for p, d in self._json_rows(parts):
                by_msg.setdefault(p["message_id"], []).append(d)
        out = []
        for m, d in self._json_rows(msgs):
            msg = dict(d, id=m["id"], type=m["type"])
            if by_msg.get(m["id"]) and not msg.get("content"):
                msg["content"] = by_msg[m["id"]]
            out.append(msg)
        return out

    def prompt_count(self, path):
        """From the export, or, for a row reference, counted in the database: the session's
        `user` messages that carry text."""
        db, sid = _split_ref(path)
        if not db:
            return _opencode.opencode_prompt_count(path)
        try:
            con = _ro(db)
            try:
                table, _ = self._session_row(con, sid)
                if table == "session_v2":
                    row = con.execute(
                        "SELECT count(*) AS n FROM session_message WHERE session_id = ? AND "
                        "type = 'user' AND trim(coalesce(json_extract(data, '$.text'), '')) != ''",
                        (sid,)).fetchone()
                elif table == "session":
                    row = con.execute(
                        "SELECT count(DISTINCT m.id) AS n FROM message m JOIN part p ON "
                        "p.message_id = m.id WHERE m.session_id = ? AND "
                        "json_extract(m.data, '$.role') = 'user' AND "
                        "json_extract(p.data, '$.type') = 'text' AND "
                        "trim(coalesce(json_extract(p.data, '$.text'), '')) != ''",
                        (sid,)).fetchone()
                else:
                    row = None
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
