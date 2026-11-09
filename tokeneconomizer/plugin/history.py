"""Importing a repository's sessions from the agents' own session logs: before the plugin, and
from the other agents afterwards.

The plugin learns from sessions it sees end. Every agent it reads (`trace.sources.SOURCES`:
Claude Code, Codex, the Copilot CLI, Pi, Grok Build, Hermes, opencode, Factory Droid) keeps its
sessions' logs, so three more kinds of session can be folded too:

  * this agent's own sessions from before the plugin was installed (a **backfill**), once, so
    learning starts at install rather than some sessions later;
  * the other agents' sessions in the same repository, before and after install (**foreign**
    sessions): a fact about the repository ("`pytest` is missing here; `python -m pytest`
    works") is true whichever agent found it, so each agent's store learns micro-skills from
    them. Tool pruning never counts a foreign session (tool names differ per agent), and the
    loop guard's envelope is seeded from this agent's own history only (turn rhythm differs);
  * every repository the logs name, at install (`te history import --all`), not only the one
    the first session starts in.

`te history import` does it for one repository; with `history_import` on (the default) the
first session start in a repository starts it by itself, in a detached process, and later
starts catch up on the other agents' sessions since (`CATCHUP_EVERY_S` apart, at most).

Which logs are this repository's: each log's recorded directory is keyed with `repo_identity`,
as a live session's is. Where that directory no longer exists (a moved checkout, a renamed
home), Claude Code and Droid file the log under a folder derived from the directory's path, so
that folder is matched EXACTLY against the one derived from the repository's root: never a
prefix, or `widget-shell` would be taken for `widget`. Left out: a log written in the last
`fold.STALE_S` (its session may still be running), one already folded or recorded by the
plugin's own hooks, one with no prompt, one over `MAX_BYTES`, one older than `history_days`, a
one-prompt scripted run (`claude -p`, `codex exec`), whose restricted tools and long autonomous
turns say little about the person's own sessions, and one that ended at or before the store's
**watermark** (below). The newest `history_max_sessions` are kept, and folded oldest first.

Time order is the point of most of what follows. Folding treats each session as the newest so
far: a fact recent sessions no longer support would come back, and tools in use lately would
look long unused, if older sessions were folded after newer ones. So every store keeps the
watermark `history.json["through"]`: the end time of the newest session it has folded, live or
imported, own or foreign. A session that is not seen live is folded only when it ended after
the watermark; one that ended before is counted as `behind` and never folded. A store with
nothing folded has no watermark, and that is the backfill. The old rule, "import only into a
store with nothing folded", is this rule's special case, and a store the old rule marked
`skipped` now catches up on everything newer than its last fold.

A live fold (the detached `te fold` a session's end spawns) first folds the sessions that
ended before it and after the watermark, at most `CATCHUP_PER_FOLD` of them, so that another
agent's session that ended in between is not stranded behind this one (`catch_up`). What ends
within `fold.STALE_S` of a live fold can still be stranded; `te history` shows the count.

Each store imports on its own: the stores of two agents in one repository are different
directories with their own locks, and the logs they read are read-only, so two agents starting
sessions in a repository at the same moment never wait on each other. The import holds the
repository's one fold lock for its whole run, and while `history.json` says `running` or
`partial` no live fold folds (`fold.import_blocks`, checked before the lock is taken, so a
session's end never waits on the import): the session stays pending, and the import folds the
pending sessions, oldest first, once it is done. The plan -- the ordered session ids and paths
-- is saved, and a run stopped by its time budget (`BUDGET_S`) is `partial` and resumes the
same plan in the same order. The run refreshes a heartbeat; one gone stale (`HEARTBEAT_STALE_S`:
the process died) lets the next session start restart it, so live folding cannot stay blocked.
A session whose fold fails is counted as skipped and not tried again, and so is one the
import's process died on `MAX_ATTEMPTS` times (`attempt`).

An imported session is folded to learn from and nothing else (`fold.fold_locked(imported=True)`):
scored against the tools its own log shows, never today's catalog; no model call; no deny rule
held, written or given back; its ledger record and `folded.json` entry say `imported`, and a
foreign one names its `origin`. `te status` counts both apart from the sessions the plugin ran
with. A foreign session's `folded.json` key is `<source>:<id>`, so one agent's ids never
collide with another's.

Reading 15,000 log heads for every plan would be slow, so the heads are cached in
`<root>/history-cache.json`, keyed by path and valid while the log's mtime and size hold. The
cache is derived state: deleting it costs one slow scan.

`history.json` (beside `folded.json`):

    {"state": "running" | "partial" | "done" | "skipped" | "stopped", "reason",
     "plan": [{"id", "key", "source", "path", "started", "ended"}], "next", "imported",
     "origins": {source: n}, "skipped": {why: n}, "over_limit", "attempt": {"id", "n"},
     "started", "heartbeat", "finished", "noticed", "through", "caught_up"}
"""
import os
import time
import traceback

from .. import loopguard, trace
from ..trace import sources as _sources
from . import fold
from .hosts import _EXIT_RE, RESULT_CAP, claude_success_code

STATE = fold.HISTORY
LOG = "history.log"
CACHE = "history-cache.json"
BUDGET_S = 1800                    # one run's time budget; a longer import resumes next time
HEARTBEAT_STALE_S = 900            # a running import this quiet has died
MAX_BYTES = 256 * 1024 * 1024      # a log larger than this is not read
HISTORY_LOCK_S = 1.0               # a session start's wait for `history.json` (then next time)
MAX_ATTEMPTS = 2                   # runs that died on one session before it is left out
MAX_RETRIES = 3                    # runs that could not read one session before it is folded as such
CATCHUP_EVERY_S = 600              # how often a session start looks for other agents' sessions
CATCHUP_PER_FOLD = 20              # earlier sessions a live fold folds first, at most
FOLDER_MATCHED = ("claude", "droid")   # agents that file logs under a folder named for the cwd
_clock = time.monotonic            # the run's budget clock


def _record_error(root):
    try:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "errors.log"), "a", encoding="utf-8") as fh:
            fh.write(f"--- {time.strftime('%Y-%m-%dT%H:%M:%S')} pid={os.getpid()} (history)\n")
            fh.write(traceback.format_exc())
    except Exception:
        pass


def read_state(ctx):
    """`history.json` as a dict; None when there is none, or it is not a JSON object."""
    doc = ctx.store.read(STATE)
    return doc if isinstance(doc, dict) else None


def _num(v, default=0):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _stale(doc, now):
    return now - _num(doc.get("heartbeat")) >= HEARTBEAT_STALE_S


def _ledger_size(store):
    try:
        return os.path.getsize(store.path("ledger.jsonl"))
    except OSError:
        return 0


def _session_records(store):
    """The ledger's records of folded sessions. Not the loop guard's judge's `overhead`
    records: it charges those while a session runs, before anything is folded."""
    if _ledger_size(store) == 0:
        return []
    return [r for r in store.read_lines("ledger.jsonl")
            if isinstance(r, dict) and r.get("kind") == "session"]


def _nothing_folded(store):
    """Has nothing been folded into this store: no `folded.json` entry, no session record?"""
    return not (store.read("folded.json") or {}) and not _session_records(store)


# ---- the watermark ---------------------------------------------------------------------------
def watermark(ctx, doc=None):
    """The end time of the newest session folded into this store, or None for a store with
    nothing folded. `history.json["through"]` where it says; else, once, the newest end (or
    fold time) the ledger's session records hold."""
    doc = read_state(ctx) if doc is None else doc
    if isinstance(doc, dict) and _num(doc.get("through"), None) is not None:
        return float(doc["through"])
    ends = [_num(r.get("ended"), None) or _num(r.get("folded_at"), None)
            for r in _session_records(ctx.store)]
    ends = [e for e in ends if e is not None]
    return max(ends) if ends else None


def advance(ctx, ended):
    """A session that ended at `ended` was folded into this store: raise its watermark. Under
    the `history` lock; never lowers it, and writes only the one key when there is no run."""
    ended = _num(ended, None)
    if ended is None:
        return

    def raise_mark(doc):
        cur = _num(doc.get("through"), None)
        if cur is not None and cur >= ended:
            return False
        doc["through"] = ended
        return None
    _update(ctx, raise_mark, timeout=10.0)


# ---- which logs ------------------------------------------------------------------------------
def _identity(cwd, env, cache):
    if cwd not in cache:
        from .paths import repo_identity
        try:
            cache[cwd] = repo_identity(cwd, env).key
        except (OSError, ValueError):
            cache[cwd] = None
    return cache[cwd]


def is_this_repository(ctx, folder, cwd, cache=None, source="claude"):
    """Is a log filed under `folder` that names `cwd` a session of this repository? Its recorded
    directory decides, keyed as a live session's is; only where that directory is gone, and the
    agent files logs by directory (`FOLDER_MATCHED`), an exact match of the folder with the one
    derived from the repository's root."""
    cache = {} if cache is None else cache
    if cwd and os.path.isdir(cwd):
        return _identity(cwd, ctx.env, cache) == ctx.repo.key
    if source not in FOLDER_MATCHED or not folder:
        return False
    root = ctx.repo.root
    names = {trace.claude_project_folder(root), trace.claude_project_folder(os.path.realpath(root))}
    return folder in names


def _enabled_sources(ctx):
    words = [str(w).strip().lower() for w in (ctx.config.history_sources or ["all"])]
    if "self" in words:
        own = _sources.SOURCES.get(ctx.host.name)
        return [own] if own else []
    return _sources.enabled(words)


def _key_for(ctx, source, sid):
    return sid if source.name == ctx.host.name else f"{source.name}:{sid}"


_EMPTY_HEAD = {"session_id": None, "cwd": None, "started": None, "headless": False,
               "folder": None}


class _Heads:
    """The head cache: `{path: {"mtime", "size", "head"}}` in `<root>/history-cache.json`.
    Read once per process, written back once, best effort; a cache that cannot be read or
    written only costs the reads it would have saved."""

    def __init__(self, root):
        from .store import Store
        self.store = Store(root)
        doc = self.store.read(CACHE)
        self.doc = doc if isinstance(doc, dict) else {}
        self.dirty = False
        self.present = set()

    def _entry(self, log):
        path = log["path"]
        hit = self.doc.get(path)
        if not (isinstance(hit, dict) and hit.get("mtime") == log["mtime"]
                and hit.get("size") == log["size"]):
            hit = self.doc[path] = {"mtime": log["mtime"], "size": log["size"]}
            self.dirty = True
        return hit

    def head(self, source, log):
        """The log's head, read once per (path, mtime, size). A reader that could not answer
        (an exception, or None: a locked database, a file mid-write) is not cached and is asked
        again next time; an empty head stands in for this pass."""
        hit = self._entry(log)
        if not isinstance(hit.get("head"), dict):
            try:
                head = source.head(log["path"])
            except Exception:
                head = None
            if not isinstance(head, dict):
                return dict(_EMPTY_HEAD)
            hit["head"] = head
            self.dirty = True
        return hit["head"]

    def prompts(self, source, log):
        hit = self._entry(log)
        if "prompts" not in hit:
            try:
                n = source.prompt_count(log["path"])
            except Exception:
                return None                      # unknown this pass; asked again next time
            hit["prompts"] = n
            self.dirty = True
        return hit["prompts"]

    def save(self, present):
        if not self.dirty:
            return
        try:
            self.store.write(CACHE, {p: v for p, v in self.doc.items() if p in present})
        except OSError:
            pass
        self.dirty = False


def scan(ctx, heads):
    """Every enabled source's logs with their heads: [(source, log, head)]. Reads, and writes
    nothing but the cache `heads`."""
    out, present = [], set()
    for source in _enabled_sources(ctx):
        extra = ctx.config.history_dirs if source.name == "claude" else ()
        try:
            logs = source.logs(ctx.env, extra)
        except Exception:
            _record_error(ctx.root)
            continue
        for log in logs:
            present.add(log["path"])
            out.append((source, log, heads.head(source, log)))
    heads.present = present
    return out


def plan(ctx, now=None, heads=None, cache=None):
    """What an import would fold here, and why the rest is left out; reads, never writes
    (but the head cache). `heads` and `cache` (repository identities by directory) may be
    shared by a caller that plans several stores of one host.

    The newest `history_max_sessions` are taken, this agent's own and the other agents'
    counted separately, so that foreign sessions never crowd out the own ones that alone feed
    tool pruning and the envelope. A session an earlier run could not read (`retry` in
    `history.json`) is planned again although the watermark has passed it.

    Returns {"sessions": [{"id", "key", "source", "path", "size", "started", "ended"}] oldest
    first, "skipped": {why: n}, "over_limit": n}."""
    now = time.time() if now is None else now
    cfg = ctx.config
    folded = ctx.store.read("folded.json") or {}
    through = watermark(ctx)
    state = read_state(ctx) or {}
    retrying = state.get("retry") if isinstance(state.get("retry"), dict) else {}
    skipped, found = {}, []
    cache = {} if cache is None else cache
    heads = _Heads(ctx.root) if heads is None else heads

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1
    for source, log, head in scan(ctx, heads):
        folder = head.get("folder") or log.get("folder")
        if not is_this_repository(ctx, folder, head.get("cwd"), cache, source=source.name):
            continue
        sid = trace.claude_safe_session_id(log["id"])
        mtime, size = log["mtime"], log["size"]
        if sid is None:
            skip("unsafe_id")
            continue
        key = _key_for(ctx, source, sid)
        if mtime > now - fold.STALE_S:
            skip("recent")                         # may be running
        elif mtime < now - cfg.history_days * 86400:
            skip("too_old")
        elif size > MAX_BYTES:
            skip("too_large")
        elif key in folded:
            skip("already_folded")
        elif (source.name == ctx.host.name
              and os.path.exists(ctx.store.path("sessions", f"{sid}.json"))):
            skip("recorded_live")
        elif through is not None and mtime <= through and key not in retrying:
            skip("behind")
        else:
            prompts = heads.prompts(source, log)
            if prompts is not None and prompts <= 0:
                skip("no_prompt")
            elif head.get("headless") and (prompts or 1) <= 1:
                skip("headless")
            else:
                found.append({"id": sid, "key": key, "source": source.name, "path": log["path"],
                              "size": size, "started": head.get("started") or mtime,
                              "ended": mtime})
    heads.save(heads.present)
    order = (lambda s: (s["ended"], s["started"], s["source"], s["id"]))
    found.sort(key=order)
    cap = max(0, int(cfg.history_max_sessions))
    own = [s for s in found if s["source"] == ctx.host.name]
    foreign = [s for s in found if s["source"] != ctx.host.name]
    over = max(0, len(own) - cap) + max(0, len(foreign) - cap)
    kept = own[max(0, len(own) - cap):] + foreign[max(0, len(foreign) - cap):]
    kept.sort(key=order)
    return {"sessions": kept, "skipped": skipped, "over_limit": over}


def _item_source(item):
    return _sources.SOURCES.get(str(item.get("source") or "claude"))


def _still_eligible(ctx, item):
    """Why a planned session is no longer to be imported (a resumed plan), or None. A plan is
    state the plugin wrote, but it is read back as data: the session id must be safe to name a
    file, and the path must be the log of that very session."""
    if not isinstance(item, dict):
        return "unsafe_id"
    sid = trace.claude_safe_session_id(item.get("id"))
    source = _item_source(item)
    path = item.get("path")
    if sid is None or source is None or not isinstance(path, str):
        return "unsafe_id"
    # A database row is `<db>#<id>`: the fragment, which names the export file, must be this
    # very id; a file must at least carry the id in its path.
    if ("#" in path and path.rpartition("#")[2] != sid) or ("#" not in path and sid not in path):
        return "unsafe_id"
    key = _key_for(ctx, source, sid)
    if key in (ctx.store.read("folded.json") or {}):
        return "already_folded"
    if source.name == ctx.host.name and os.path.exists(ctx.store.path("sessions", f"{sid}.json")):
        return "recorded_live"
    if "#" not in path:
        try:
            if os.path.getsize(path) > MAX_BYTES:
                return "too_large"
        except OSError:
            return "gone"
    elif _num(item.get("size")) > MAX_BYTES:
        return "too_large"
    return None


# ---- the loop guard's envelope ---------------------------------------------------------------
def replay_envelope(ctx, transcript_path, source=None):
    """Seed the loop guard's envelope from an imported session of this agent, as its Stop hooks
    would have. Only a source that splits its log into prompt-turns (`turns`) can; a foreign
    session never does, since turn rhythm differs between agents.

    Each prompt-turn is folded call by call into a fresh `TurnGuard` with no envelope and no
    judge, so nothing can redirect or stop it, and the result text and exit code it gets are the
    ones a live hook would have (a failed call's "Exit code N", a successful shell call's 0 where
    `claude_success_code` gives it, else none). A turn feeds the envelope, as a live one does,
    when it stopped normally: not interrupted by the person, and, for the session's last turn,
    answered (a log cut off mid-turn did not stop normally).
    Returns the number of turns recorded."""
    cfg = ctx.config
    source = _sources.SOURCES.get(ctx.host.name) if source is None else source
    if cfg.loopguard == "off" or source is None or source.name != ctx.host.name:
        return 0
    turns_of = getattr(source, "turns", None)
    if turns_of is None:
        return 0
    level, overrides = cfg.loopguard_aggr, cfg.loopguard_thresholds
    thr = loopguard.base_thresholds(level, overrides)
    recorded = 0
    turns = turns_of(transcript_path)
    turn = next(turns, None)
    while turn is not None:
        following = next(turns, None)
        guard = loopguard.TurnGuard()
        for name, args, result, ok, _cwd in trace.claude_calls(turn):
            result = (result or "")[:RESULT_CAP]
            if ok:
                code = claude_success_code(name, args)
            else:
                m = _EXIT_RE.search(result[:400])
                code = int(m.group(1)) if m else None
            guard.fold(name, args if isinstance(args, dict) else {}, result, code,
                       envelope=None, thr=thr, judge=None)
        normal = not trace.claude_turn_interrupted(turn) and (
            following is not None or trace.claude_turn_answered(turn))
        if guard.cum_turns and normal:
            loopguard.record_turn(ctx.store, guard, level=level, overrides=overrides)
            recorded += 1
        turn = following
    return recorded


# ---- the run ---------------------------------------------------------------------------------
def _update(ctx, change, timeout=10.0):
    """Read-modify-write `history.json` under its own lock; returns the new doc, or None when
    the lock was busy. `change(doc)` returns False to write nothing."""
    with ctx.store.locked("history", timeout=timeout) as got:
        if not got:
            return None
        doc = read_state(ctx) or {}
        if change(doc) is False:
            return doc
        ctx.store.write(STATE, doc)
        return doc


def _save(ctx, doc):
    with ctx.store.locked("history", timeout=10.0):
        # Another writer (`advance`, a fold) may have raised the watermark meanwhile.
        cur = read_state(ctx) or {}
        mark = max(_num(cur.get("through"), 0), _num(doc.get("through"), 0))
        if mark:
            doc["through"] = mark
        ctx.store.write(STATE, doc)


def _say(log, text):
    if log is not None:
        try:
            log(text)
        except Exception:
            pass


def run(ctx, auto=False, now=None, budget_s=BUDGET_S, log=None, heads=None, cache=None):
    """Import this repository's earlier and foreign sessions into this store, holding the fold
    lock throughout.

    `auto`: started by a session start, which runs only while `history.json` says it should
    (`running` or `partial`) and `history_import` is on. Returns {"state", "imported",
    "skipped", "reason"}: state "busy" when the fold lock stayed taken."""
    now = time.time() if now is None else now
    with ctx.store.locked("fold", timeout=fold.FOLD_LOCK_WAIT_S) as got:
        if not got:
            return {"state": "busy", "reason": "another fold holds this repository's lock"}
        doc = read_state(ctx) or {}
        state = doc.get("state")
        blocking = state in fold.HISTORY_BLOCKING
        # A plan stopped part-way resumes in order: by itself while it blocks live folds, and
        # by hand after `history_import=off` stopped it (`te history import` marks the store
        # `running` first, so a by-hand run arrives here with its plan, if any, under that).
        resuming = ((blocking or (not auto and state == "stopped"))
                    and isinstance(doc.get("plan"), list) and bool(doc.get("plan")))
        if auto and not blocking:
            return {"state": state or "none", "reason": "nothing to resume"}
        if auto and not ctx.config.history_import:
            doc.update(state="stopped", reason="history_import is off", at=time.time())
            _save(ctx, doc)
            return {"state": "stopped", "reason": doc["reason"]}
        try:
            return _run_locked(ctx, doc, resuming, now, budget_s, log, heads, cache)
        except Exception:
            # Not left `running`: live folds would wait for a run that fails again each time.
            _record_error(ctx.root)
            doc = read_state(ctx) or {}
            doc.update(state="stopped", reason="the import failed (see errors.log)",
                       at=time.time())
            _save(ctx, doc)
            return {"state": "error", "reason": doc["reason"]}


def _run_locked(ctx, doc, resuming, now, budget_s, log, heads=None, cache=None):
    """`run` once it may go on, under the fold lock: plan (or take up the saved plan), fold
    each session in order, save the place after each, then fold the sessions left pending."""
    before = int(_num(doc.get("imported")))
    if not resuming:
        # Live folds wait from now on, while the plan is made, not only once it is. What
        # earlier runs read (`imported`, `origins`, `retry`) and announced stands; the plan is
        # new.
        base = {k: v for k, v in doc.items()
                if k not in ("plan", "next", "attempt", "skipped", "over_limit", "planned")}
        _save(ctx, dict(base, state="running", heartbeat=time.time()))
        p = plan(ctx, now, heads=heads, cache=cache)
        doc = dict(base, state="running", plan=p["sessions"], next=0, skipped=p["skipped"],
                   over_limit=p["over_limit"], planned=len(p["sessions"]), started=time.time())
        doc.setdefault("imported", 0)
        doc.setdefault("noticed", True)
        if p["sessions"]:
            doc["noticed"] = False
            _say(log, f"{len(p['sessions'])} earlier session(s) of {ctx.repo.label} to read")
    doc.update(state="running", pid=os.getpid(), heartbeat=time.time())
    doc["runs"] = int(_num(doc.get("runs"))) + 1
    _save(ctx, doc)
    skipped = doc["skipped"] = dict(doc["skipped"] if isinstance(doc.get("skipped"), dict)
                                    else {})
    origins = doc["origins"] = dict(doc["origins"] if isinstance(doc.get("origins"), dict)
                                    else {})
    items = doc["plan"] if isinstance(doc.get("plan"), list) else []
    deadline = _clock() + max(0.0, float(budget_s))
    i = int(_num(doc.get("next")))
    while i < len(items):
        if _clock() >= deadline:
            break
        _import_one(ctx, doc, items[i], skipped, origins, log)
        i += 1
        doc.pop("attempt", None)
        doc.update(next=i, heartbeat=time.time())
        _save(ctx, doc)
    if i < len(items):
        doc["state"] = "partial"            # the next session start resumes the plan
    else:
        doc.update(state="done", finished=time.time(), caught_up=time.time())
        doc.pop("plan", None)
    _save(ctx, doc)
    if doc["state"] == "done":
        # The sessions that ended while the import ran, in the order they ran.
        for sid in fold.pending_sessions(ctx):
            try:
                fold.fold_locked(ctx, sid)
            except Exception:
                _record_error(ctx.root)
    return {"state": doc["state"], "imported": doc.get("imported", 0),
            "read": int(_num(doc.get("imported"))) - before,
            "skipped": dict(skipped), "over_limit": doc.get("over_limit", 0),
            "left": len(items) - i, "reason": None}


def _fold_item(ctx, item, source, final=False):
    """Fold one planned session into this store, as an imported one of `source`; the ledger
    record, None when it was folded already, or `{"retry": True, ...}` when its log could not
    be read this time and `final` is False (nothing written: it is planned again)."""
    key = _key_for(ctx, source, item["id"])
    return fold.fold_locked(ctx, key, item["path"], imported=True,
                            span=(item.get("started"), item.get("ended")), source=source,
                            final=final)


def _retries(doc, key, n=None):
    """The retry count of `key` in a history doc; with `n`, set it (0 removes it)."""
    retry = doc.get("retry") if isinstance(doc.get("retry"), dict) else {}
    if n is not None:
        if n > 0:
            retry[key] = n
        else:
            retry.pop(key, None)
        if retry:
            doc["retry"] = retry
        else:
            doc.pop("retry", None)
    return int(_num(retry.get(key)))


def _import_one(ctx, doc, item, skipped, origins, log):
    """Fold one planned session, under the fold lock. `doc["attempt"]`, saved before the fold
    and dropped with the place saved after it, tells a restarted run what the last one was
    doing when its process died: a session it died on `MAX_ATTEMPTS` times is left out (an
    error), so that no log can block live folds for good; one it folded before it could save
    its place is counted, and its turns replayed into the envelope."""
    sid = item.get("id") if isinstance(item, dict) else None
    source = _item_source(item) if isinstance(item, dict) else None
    att = doc.get("attempt") if isinstance(doc.get("attempt"), dict) else {}
    tries = int(_num(att.get("n"))) if sid is not None and att.get("id") == sid else 0
    why = _still_eligible(ctx, item)
    key = _key_for(ctx, source, sid) if source and sid else None
    if why == "already_folded" and key and _folded_by_import(ctx.store, key):
        # This import folded it, and its process died before it saved its place.
        doc["imported"] = int(_num(doc.get("imported"))) + 1
        origins[source.name] = int(origins.get(source.name, 0)) + 1
        if tries < MAX_ATTEMPTS:
            doc["attempt"] = {"id": sid, "n": tries + 1}
            _save(ctx, doc)
            _replay(ctx, item, source)
        return
    if why:
        skipped[why] = skipped.get(why, 0) + 1
        return
    if tries >= MAX_ATTEMPTS:
        skipped["error"] = skipped.get("error", 0) + 1      # the process died on it each time
        return
    doc["attempt"] = {"id": sid, "n": tries + 1}
    _save(ctx, doc)
    # A log that could not be read (a locked database, a file mid-write) is planned again at
    # the next run, `MAX_RETRIES` times, before it is folded as unreadable for good.
    retried = _retries(doc, key)
    try:
        rec = _fold_item(ctx, item, source, final=retried >= MAX_RETRIES)
        if rec is not None and rec.get("retry"):
            _retries(doc, key, retried + 1)
            skipped["retry_later"] = skipped.get("retry_later", 0) + 1
        elif rec is not None and not rec.get("error"):
            _retries(doc, key, 0)
            doc["imported"] = int(_num(doc.get("imported"))) + 1
            origins[source.name] = int(origins.get(source.name, 0)) + 1
            _replay(ctx, item, source)
            _say(log, f"read {key}")
        elif rec is not None:
            _retries(doc, key, 0)
            skipped["unreadable"] = skipped.get("unreadable", 0) + 1
    except Exception:
        _record_error(ctx.root)
        skipped["error"] = skipped.get("error", 0) + 1


def _replay(ctx, item, source):
    """The envelope's share of an imported session of this agent (`replay_envelope`); a
    failure there costs the envelope, never the import."""
    try:
        replay_envelope(ctx, item["path"], source)
    except Exception:
        _record_error(ctx.root)


def _folded_by_import(store, key):
    """Is `key` in `folded.json` as a session this repository's import folded?"""
    entry = (store.read("folded.json") or {}).get(key)
    return isinstance(entry, dict) and bool(entry.get("imported"))


# ---- a live fold's catch-up ------------------------------------------------------------------
def catch_up(ctx, before, limit=CATCHUP_PER_FOLD):
    """Under the fold lock, before a live session that ended at `before` is folded: fold the
    imported-eligible sessions that ended before it (and after the watermark), oldest first, at
    most `limit`. Nothing while an import runs or `history_import` is off. Returns how many."""
    if not ctx.config.history_import or fold.import_blocks(ctx):
        return 0
    try:
        p = plan(ctx)
    except Exception:
        _record_error(ctx.root)
        return 0
    items = [s for s in p["sessions"] if s["ended"] < _num(before, float("inf"))][:limit]
    done = []
    state = read_state(ctx) or {}
    retry_later, settled = [], []
    for item in items:
        source = _item_source(item)
        if source is None:
            continue
        key = _key_for(ctx, source, item["id"])
        try:
            rec = _fold_item(ctx, item, source, final=_retries(state, key) >= MAX_RETRIES)
        except Exception:
            _record_error(ctx.root)
            continue
        if rec is not None and rec.get("retry"):
            retry_later.append(key)
        elif rec is not None:
            settled.append(key)
            if not rec.get("error"):
                done.append(item)
                _replay(ctx, item, source)
    if done or retry_later or settled:
        def count(doc):
            origins = doc.get("origins") if isinstance(doc.get("origins"), dict) else {}
            for item in done:
                origins[item["source"]] = int(origins.get(item["source"], 0)) + 1
            for key in retry_later:
                _retries(doc, key, _retries(doc, key) + 1)
            for key in settled:
                _retries(doc, key, 0)
            doc["origins"] = origins
            doc["imported"] = int(_num(doc.get("imported"))) + len(done)
            if done:
                doc["noticed"] = False      # the next session start says what was read
            doc.setdefault("state", "done")
            return None
        _update(ctx, count)
    return len(done)


# ---- what a session start does ---------------------------------------------------------------
def maybe_start(ctx, now=None):
    """At a session start: start the import (the backfill, or a catch-up on the other agents'
    sessions `CATCHUP_EVERY_S` after the last), resume one stopped by its budget, or restart
    one whose process died -- in a detached `te history import` (inline when TE_FOLD_SYNC=1,
    for tests). Reads only `history.json`."""
    now = time.time() if now is None else now
    exists = os.path.exists(ctx.store.path(STATE))
    if exists and read_state(ctx) is None:
        return None                     # unreadable: left alone (and it blocks no fold)
    if not exists and not ctx.config.history_import:
        return None
    start = {}

    def decide(doc):
        state = doc.get("state")
        if state in fold.HISTORY_BLOCKING and not ctx.config.history_import:
            doc.update(state="stopped", reason="history_import is off", at=now)
            return None
        if not ctx.config.history_import:
            return False
        if state == "partial" or (state == "running" and _stale(doc, now)):
            doc["heartbeat"] = now      # so that another session start does not start one too
        elif state == "running":
            return False                # alive: left to run
        elif state in (None, "done", "skipped", "stopped"):
            if state is not None and now - _num(doc.get("caught_up")) < CATCHUP_EVERY_S:
                return False
            doc.update(state="running", heartbeat=now)
            doc.setdefault("started", now)
            doc.setdefault("noticed", False)
            if state == "skipped":
                doc.pop("reason", None)
        else:
            return False
        start["go"] = True
        return None
    if _update(ctx, decide, timeout=HISTORY_LOCK_S) is None or not start:
        return None
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        return run(ctx, auto=True)
    return ctx.spawn(["history", "import", "--auto"], LOG)


def _labels(origins):
    names = [_sources.SOURCES[s].label for s in sorted(origins or {})
             if s in _sources.SOURCES and origins[s]]
    if not names:
        return "the agents' history"
    if len(names) == 1:
        return f"{names[0]}'s history"
    return ", ".join(names[:-1]) + f" and {names[-1]} history"


def notice(ctx):
    """Once, after an import finished: the line saying it read the repository's history."""
    doc = read_state(ctx)
    if not doc or doc.get("state") != "done" or doc.get("noticed"):
        return None
    out = {}

    def mark(d):
        if d.get("state") != "done" or d.get("noticed"):
            return False
        d["noticed"] = True
        total = int(_num(d.get("imported")))
        out["n"] = total - int(_num(d.get("announced")))       # what is new since last said
        d["announced"] = total
        out["origins"] = d.get("origins") if isinstance(d.get("origins"), dict) else {}
        return None
    if _update(ctx, mark, timeout=HISTORY_LOCK_S) is None or out.get("n", 0) <= 0:
        return None
    return (f"TokenEconomizer read {out['n']} earlier session(s) of {ctx.repo.label} from "
            f"{_labels(out['origins'])}.")


def summary(ctx):
    """`history.json` without its plan, for `te status` and `te history`."""
    doc = read_state(ctx)
    if doc is None:
        return None
    out = {k: v for k, v in doc.items() if k != "plan"}
    if isinstance(doc.get("plan"), list):
        out["left"] = max(0, len(doc["plan"]) - int(_num(doc.get("next"))))
    return out


def describe(doc):
    """One line for `te status`: what the import did here."""
    if not doc:
        return None
    state = doc.get("state")
    n = int(_num(doc.get("imported")))
    origins = doc.get("origins") if isinstance(doc.get("origins"), dict) else {}
    if state == "done":
        return f"read {n} earlier session(s) from {_labels(origins)}"
    if state in fold.HISTORY_BLOCKING:
        return (f"reading the agents' history: {n} session(s) so far, {doc.get('left', '?')} "
                f"to go; sessions ending meanwhile are folded after it")
    if state == "skipped":
        return f"not read ({doc.get('reason') or 'skipped'})"
    if state == "stopped":
        return f"stopped after {n} session(s) ({doc.get('reason') or 'stopped'})"
    return None


# ---- every repository ------------------------------------------------------------------------
def repositories(root, env, cfg, heads=None):
    """Every repository the enabled sources' logs name, with the agents that ran there:
    {repo_key: {"repo": RepoId, "hosts": {name}, "newest": ended, "sessions": n}}. Logs whose
    directory is gone, or outside any checkout, are not placed."""
    from . import hosts as _hosts
    from .paths import repo_identity

    class _Ctx:            # the little `scan` needs: where to find things, nothing else
        pass
    ctx = _Ctx()
    ctx.root, ctx.env, ctx.config = root, env, cfg
    ctx.host = _hosts.HOSTS["claude"]()
    heads = _Heads(root) if heads is None else heads
    groups, ids = {}, {}
    entries = scan(ctx, heads)
    heads.save(heads.present)
    for source, log, head in entries:
        cwd = head.get("cwd")
        if not (isinstance(cwd, str) and os.path.isdir(cwd)):
            continue
        if cwd not in ids:
            try:
                ids[cwd] = repo_identity(cwd, env)
            except (OSError, ValueError):
                ids[cwd] = None
        repo = ids[cwd]
        if repo is None or repo.how == "dir":
            continue
        g = groups.setdefault(repo.key, {"repo": repo, "hosts": set(), "newest": 0.0,
                                         "sessions": 0})
        if source.name in _hosts.HOSTS:
            g["hosts"].add(source.name)
        g["newest"] = max(g["newest"], log["mtime"])
        g["sessions"] += 1
    return groups


def run_all(root, env, cfg, dry_run=False, log=None, build=None):
    """`te history import --all`: for every repository the logs name, import into the store of
    each agent that ran there (the other agents' stores are made at their first session start,
    and import then). Repositories with the newest sessions first. Returns
    {"repos": n, "stores": n, "imported": n, "dry_run"}."""
    from . import hosts as _hosts
    from .context import LiveContext
    build = build or (lambda host, cwd: LiveContext.build(host, cwd, env))
    # One head cache and one identity cache for every store of the host: the logs are read
    # once, not once per store.
    heads, cache = _Heads(root), {}
    groups = repositories(root, env, cfg, heads=heads)
    order = sorted(groups.values(), key=lambda g: -g["newest"])
    totals = {"repos": len(order), "stores": 0, "imported": 0, "dry_run": dry_run}
    for g in order:
        for name in sorted(g["hosts"]):
            totals["stores"] += 1
            ctx = build(_hosts.HOSTS[name](), g["repo"].root)
            if dry_run:
                p = plan(ctx, heads=heads, cache=cache)
                _say(log, f"{g['repo'].label} [{name}]: {len(p['sessions'])} session(s) would "
                          f"be read" + (f", left out: " + ", ".join(
                              f"{k} {v}" for k, v in sorted(p["skipped"].items()))
                                        if p["skipped"] else ""))
                continue
            doc = read_state(ctx) or {}
            if doc.get("state") in fold.HISTORY_BLOCKING and not _stale(doc, time.time()):
                _say(log, f"{g['repo'].label} [{name}]: an import is running; left to it")
                continue
            # A saved plan (a run stopped by its budget) is kept and resumed; else a new one.
            _save(ctx, dict(doc, state="running", heartbeat=time.time()))
            res = run(ctx, auto=True, log=log, heads=heads, cache=cache)
            if res.get("state") == "busy":
                _save(ctx, doc)             # not left `running` by a run that never began
            totals["imported"] += int(_num(res.get("read")))
            _say(log, f"{g['repo'].label} [{name}]: {res['state']}, {res.get('read', 0)} "
                      f"session(s) read")
    return totals
