"""Factory Droid's session log, read as history: prompts, calls and results, as Claude-shaped
lines. The fixture is synthetic, in the shape Droid 1.5.0 writes."""
import json
import os

from tokeneconomizer import trace
from tokeneconomizer.trace import droid


def write_droid_log(sessions_dir, cwd, sid, entries):
    folder = trace.claude_project_folder(cwd)
    d = os.path.join(sessions_dir, folder)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sid}.jsonl")
    lines = [{"type": "session_start", "id": sid, "title": "New Session", "owner": "u",
              "version": 2, "cwd": cwd, "hostId": "h"}]
    lines.extend(entries)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def user(eid, content, ts):
    return {"type": "message", "id": eid, "timestamp": ts,
            "message": {"role": "user", "content": content}}


def assistant(eid, blocks, ts):
    return {"type": "message", "id": eid, "timestamp": ts,
            "message": {"role": "assistant", "content": blocks,
                        "modelId": "custom:native-claude-opus-5-5-0", "reasoningEffort": "high"}}


ENTRIES = [
    user("context-1", [{"type": "text", "text": "<system-reminder>catalog</system-reminder>"}],
         "2026-11-03T11:53:46.818Z"),
    user("p1", [{"type": "text", "text": "fix the tests"}], "2026-11-03T11:53:46.850Z"),
    assistant("a1", [{"type": "thinking", "thinking": "..."},
                     {"type": "tool_use", "id": "toolu_1", "name": "Execute",
                      "input": {"command": "pytest -q"}}], "2026-11-03T11:53:50.000Z"),
    user("r1", [{"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "Exit code 127\nbash: pytest: command not found"}],
         "2026-11-03T11:53:51.000Z"),
    assistant("a2", [{"type": "tool_use", "id": "toolu_2", "name": "Execute",
                      "input": {"command": "python -m pytest -q"}}], "2026-11-03T11:53:55.000Z"),
    user("repl-tool-result:x-1", [{"type": "tool_result", "tool_use_id": "toolu_2",
                                    "is_error": False,
                                    "content": [{"type": "text", "text": "3 passed"}]}],
         "2026-11-03T11:53:56.000Z"),
    {"type": "agent_turn_outcome", "turnId": "p1", "reason": "completed", "resultKind": "text"},
    assistant("a3", [{"type": "text", "text": "Done."}], "2026-11-03T11:54:00.000Z"),
    {"type": "todo_state", "todos": []},
]


def test_head_prompts_and_calls(tmp_path):
    path = write_droid_log(str(tmp_path / "sessions"), "/work/widget", "s1", ENTRIES)
    head = droid.droid_log_head(path)
    assert head["session_id"] == "s1" and head["cwd"] == "/work/widget"
    assert head["started"] is not None and head["ended"] >= head["started"]
    assert droid.droid_prompt_count(path) == 1
    out = str(tmp_path / "trace.jsonl")
    assert droid.droid_session_to_claude(path, out) == 2
    calls = trace.claude_calls(trace.jsonl.lines(out))
    assert [(n, a["command"], ok, cwd) for n, a, _, ok, cwd in calls] == [
        ("Bash", "pytest -q", True, "/work/widget"),
        ("Bash", "python -m pytest -q", True, "/work/widget")]
    assert "command not found" in calls[0][2]
    assert trace.claude_prompt_count(out) == 1
    assert len(list(trace.claude_prompt_turns(out))) == 1
    assert droid.droid_session_usage(path)["model"] == "custom:native-claude-opus-5-5-0"
    assert droid.droid_session_usage(path)["input_tokens"] is None


def test_the_logs_droid_keeps(tmp_path):
    root = str(tmp_path / "sessions")
    write_droid_log(root, "/work/a", "s1", ENTRIES)
    write_droid_log(root, "/work/b", "s2", ENTRIES[:2])
    logs = droid.droid_history_logs(root)
    assert [(f, os.path.basename(p)) for f, p, _, _ in logs] == [
        ("-work-a", "s1.jsonl"), ("-work-b", "s2.jsonl")]
    assert droid.droid_history_logs(str(tmp_path / "none")) == []
    assert droid.droid_sessions_dir({"HOME": "/h"}) == "/h/.factory/sessions"
    assert droid.droid_sessions_dir({"FACTORY_HOME": "/f"}) == "/f/sessions"


def test_a_log_without_conversation_converts_to_nothing(tmp_path):
    path = write_droid_log(str(tmp_path / "sessions"), "/work/c", "s3", [])
    assert droid.droid_session_to_claude(path, str(tmp_path / "t.jsonl")) is None
    assert droid.droid_prompt_count(path) == 0


def test_droids_tools_in_claude_codes_vocabulary():
    assert droid.droid_tool("Execute", {"summary": "s", "command": "ls"}) == ("Bash",
                                                                               {"command": "ls"})
    assert droid.droid_tool("Create", {"file_path": "/a", "content": "x"}) == (
        "Write", {"file_path": "/a"})
    assert droid.droid_tool("LS", {"directory_path": "/d"}) == ("LS", {"path": "/d"})
    assert droid.droid_tool("Glob", {"patterns": ["*.py", "*.md"], "folder": "/d"}) == (
        "Glob", {"pattern": "*.py *.md", "path": "/d"})
    assert droid.droid_tool("ApplyPatch", {"input": "*** Begin Patch\n*** Update File: a/b.py\n"
                                                    "*** Add File: c.py\n*** End Patch"}) == (
        "apply_patch", {"path": "a/b.py", "paths": ["a/b.py", "c.py"]})
    assert droid.droid_tool("Script", {"script": "x"}) == ("Script", {"script": "x"})
    assert droid.droid_tool(None, None) == ("", {})
