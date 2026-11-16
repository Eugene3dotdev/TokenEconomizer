"""Amp (ampcode CLI) threads, normalized to the Claude-shaped trace the mechanisms read.

Amp keeps no conversation on this machine (0.0.1794283240): a thread lives on Amp's server, and
`~/.cache/amp/logs/threads/` holds frame metadata, not messages. A thread reaches a file two
ways, and this module reads both:

  * the plugin's copy, `<state root>/amp-threads/<thread id>.jsonl`: a `{"type": "thread", id,
    cwd, created, mode}` header, then one `{"type": "message", ...}` line per message the
    plugin saw at `agent.end`, in the plugin API's `ThreadMessage` shape. A message can be
    written twice (a turn retried after an error); the later line wins;
  * `amp threads export <thread id>` saved as `<thread id>.json`: `{id, created (epoch ms),
    env: {initial: {workingDirectory (a file:// URI), platform: {client}}}, messages}`.

The two message shapes differ only in their tool results and ids. A plugin message is `{role,
id, content}` with `{type: "tool_result", toolUseID, status, output}`; an exported one is
`{role, messageId, content}` with `{type: "tool_result", toolUseID, run: {status, result}}`.
The shell tool's result is `{output, exitCode}` in both. Assistant messages carry `tool_use`
blocks (`{id, name, input}`) and their own `usage`.

Usage, as Amp records it per assistant message: `inputTokens` excludes the cache fields
(`totalInputTokens` is the three together), so the mapping to the ledger's four fields is
direct.

What the plugin adds to a prompt (`agent.start`'s returned message) is wrapped in
`<tokeneconomizer>` tags and left out of the prompt text here: it is the plugin's, not the
person's.
"""
import json
import os
import re
from urllib.parse import unquote, urlparse

from .calls import strip_rtk_prefix
from .jsonl import records

# Amp's tool names, as the mechanisms know them. `shell_command` is the shell on GPT-backed
# modes; `Bash`, `Read`, `Grep` and `glob` are the Claude-backed modes' tools.
_NAMES = {"shell_command": "Bash", "Bash": "Bash", "Read": "Read", "edit_file": "Edit",
          "create_file": "Write", "Grep": "Grep", "glob": "Glob", "web_search": "WebSearch",
          "read_web_page": "WebFetch", "Task": "Task"}
PLUGIN_TAG = "tokeneconomizer"
_INJECTED_RE = re.compile(rf"\s*<{PLUGIN_TAG}>.*?</{PLUGIN_TAG}>\s*", re.S)


def amp_tool(name, args):
    """(name, args) of an Amp tool call in Claude Code's vocabulary: the file and command tools
    renamed with the arguments the mechanisms read; any other tool as it is."""
    args = args if isinstance(args, dict) else {}
    mapped = _NAMES.get(name or "")
    if mapped == "Bash":
        return mapped, {"command": strip_rtk_prefix(str(args.get("command") or args.get("cmd")
                                                         or ""))}
    if mapped in ("Read", "Edit", "Write"):
        out = {"file_path": str(args.get("path") or args.get("file_path") or "")}
        if mapped == "Read" and isinstance(args.get("read_range"), list):
            out["read_range"] = args["read_range"]
        return mapped, out
    if mapped == "Grep":
        out = {"pattern": str(args.get("pattern") or ""), "path": str(args.get("path") or "")}
        if args.get("glob"):
            out["glob"] = args["glob"]
        return mapped, out
    if mapped == "Glob":
        return mapped, {"pattern": str(args.get("filePattern") or args.get("pattern") or "")}
    return mapped or name or "", args


def _uri_path(uri):
    if not isinstance(uri, str) or not uri:
        return None
    if uri.startswith("file://"):
        path = unquote(urlparse(uri).path)
        return path or None
    return uri if os.path.isabs(uri) else None


def _text(value):
    """A tool result's text: a string as it is, `{output}` its output, content blocks their
    text, anything else as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("output", "content", "text", "result"):
            if isinstance(value.get(key), (str, list)):
                return _text(value[key])
    if isinstance(value, list) and all(isinstance(b, dict) for b in value):
        texts = [b.get("text") for b in value if isinstance(b.get("text"), str)]
        if texts:
            return "\n".join(texts)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def amp_result(block):
    """(text, exit_code_or_None, failed) of a `tool_result` block in either shape."""
    run = block.get("run") if isinstance(block.get("run"), dict) else None
    if run is not None:
        status, output, error = run.get("status"), run.get("result"), run.get("error")
    else:
        status, output, error = block.get("status"), block.get("output"), block.get("error")
    code = output.get("exitCode") if isinstance(output, dict) else None
    code = code if isinstance(code, int) and not isinstance(code, bool) else None
    text = _text(output)
    if status in ("error", "cancelled") and not text:
        text = _text(error) or str(status)
    # A call a plugin refused comes back `done`, its text saying so (0.0.1794283240).
    failed = (status in ("error", "cancelled") or (code is not None and code != 0)
              or text.startswith("Tool rejected by plugin:"))
    return text, code, failed


def _load(path):
    """(head, messages) of a thread file in either shape; ({}, []) when it cannot be read.
    `head` is {id, cwd, created (epoch seconds), headless}."""
    if not path or not os.path.isfile(path):
        return {}, []
    if str(path).endswith(".jsonl"):
        head, by_id, order = {}, {}, []
        for rec in records(path):
            if rec.get("type") == "thread" and not head:
                head = _plugin_head(rec)
            elif rec.get("type") == "message" and rec.get("role") in ("user", "assistant"):
                key = rec.get("id")
                key = key if isinstance(key, (str, int)) and not isinstance(key, bool) else \
                    f"#{len(order)}"
                if key not in by_id:
                    order.append(key)
                by_id[key] = rec
        return head, [by_id[k] for k in order]
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}, []
    if not isinstance(data, dict):
        return {}, []
    msgs = data.get("messages")
    return _export_head(data), [m for m in msgs if isinstance(m, dict)
                                and m.get("role") in ("user", "assistant")] \
        if isinstance(msgs, list) else []


def _epoch(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) / 1000.0 if value > 1e11 else float(value)
    return None


def _plugin_head(rec):
    cwd = rec.get("cwd")
    return {"id": rec.get("id") if isinstance(rec.get("id"), str) else None,
            "cwd": cwd if isinstance(cwd, str) and os.path.isabs(cwd) else None,
            "created": _epoch(rec.get("created")),
            "headless": str(rec.get("mode") or "") == "execute"}


def _export_head(data):
    env = data.get("env") if isinstance(data.get("env"), dict) else {}
    initial = env.get("initial") if isinstance(env.get("initial"), dict) else {}
    platform = initial.get("platform") if isinstance(initial.get("platform"), dict) else {}
    return {"id": data.get("id") if isinstance(data.get("id"), str) else None,
            "cwd": _uri_path(initial.get("workingDirectory")),
            "created": _epoch(data.get("created")),
            "headless": "execute" in str(platform.get("client") or "").lower()}


def amp_thread_head(path):
    """{id, cwd, created, headless} of a thread file. For the plugin's copy that is its first
    line; an export is one JSON document, read whole."""
    if str(path).endswith(".jsonl"):
        rec = next(records(path), None)
        return _plugin_head(rec) if isinstance(rec, dict) and rec.get("type") == "thread" else {}
    return _load(path)[0]


def _prompt(msg):
    """The text a person typed in a user message, without what the plugin appended."""
    parts = [b.get("text") for b in msg.get("content") or []
             if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)]
    return _INJECTED_RE.sub("\n", "\n".join(parts)).strip()


def amp_prompt_count(path):
    """How many prompts the thread ran: user messages with text (a message holding only tool
    results is the agent's own)."""
    return sum(1 for m in _load(path)[1] if m.get("role") == "user" and _prompt(m))


def amp_thread_to_claude(path, out_path):
    """Write an Amp thread as Claude Code transcript lines.

    A shell call is placed in its `workdir` where it names one, else in the thread's directory.
    Returns the number of tool calls written, or None when the file holds no message. A call
    with no result yet is written without one, as a thread that ended mid-call."""
    head, messages = _load(path)
    if not messages:
        return None
    cwd, n, out = head.get("cwd"), 0, []
    for m in messages:
        content = [b for b in m.get("content") or [] if isinstance(b, dict)]
        if m.get("role") == "user":
            text = _prompt(m)
            if text:
                out.append({"type": "user", "message": {"role": "user", "content": text}})
            results = [b for b in content if b.get("type") == "tool_result"]
            for b in results:
                text, _code, failed = amp_result(b)
                out.append({"type": "user", "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.get("toolUseID"), "content": text,
                     "is_error": failed}]}})
            continue
        blocks, where = [], cwd
        for b in content:
            if b.get("type") == "tool_use":
                name, args = amp_tool(b.get("name"), b.get("input"))
                raw = b.get("input") if isinstance(b.get("input"), dict) else {}
                if name == "Bash" and isinstance(raw.get("workdir"), str) and \
                        os.path.isabs(raw["workdir"]):
                    where = raw["workdir"]
                blocks.append({"type": "tool_use", "id": b.get("id"), "name": name,
                               "input": args})
                n += 1
            elif b.get("type") == "text" and isinstance(b.get("text"), str):
                blocks.append({"type": "text", "text": b["text"]})
        msg = {"id": str(m.get("id") or m.get("messageId") or ""), "role": "assistant",
               "content": blocks}
        usage = _usage(m.get("usage"))
        if usage:
            msg["model"] = usage.pop("model")
            msg["usage"] = usage
        out.append(dict({"type": "assistant", "message": msg},
                        **({"cwd": where} if where else {})))
    with open(out_path, "w", encoding="utf-8") as fh:
        for item in out:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return n


def _usage(u):
    if not isinstance(u, dict) or u.get("outputTokens") is None:
        return None
    return {"input_tokens": int(u.get("inputTokens") or 0),
            "cache_read_input_tokens": int(u.get("cacheReadInputTokens") or 0),
            "cache_creation_input_tokens": int(u.get("cacheCreationInputTokens") or 0),
            "output_tokens": int(u.get("outputTokens") or 0),
            "model": u.get("model") if isinstance(u.get("model"), str) else None}


def amp_thread_usage(path):
    """Token breakdown of a thread, summed over its assistant messages; the model is the one
    that answered most of them. Amp states no cost per message, so none is given."""
    totals = {"input_tokens": 0, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "output_tokens": 0}
    requests, models = 0, {}
    for m in _load(path)[1]:
        u = _usage(m.get("usage")) if m.get("role") == "assistant" else None
        if not u:
            continue
        requests += 1
        model = u.pop("model")
        if model:
            models[model] = models.get(model, 0) + 1
        for k in totals:
            totals[k] += u[k]
    if not requests:
        return {"input_tokens": None, "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None, "output_tokens": None, "requests": 0,
                "model": None}
    totals.update(requests=requests, model=max(models, key=models.get) if models else None)
    return totals
