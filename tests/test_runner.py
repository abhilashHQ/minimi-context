from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from runner import run_questions as rq

FAKE_CLAUDE = textwrap.dedent(
    """\
    import json, os, sys
    from pathlib import Path

    argv = sys.argv[1:]
    log = Path(os.environ["FAKE_CLAUDE_LOG"])
    config = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text())
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"argv": argv, "cwd": os.getcwd(),
                            "cwd_files": os.listdir("."), "config": config}) + "\\n")
    question = argv[argv.index("-p") + 1]
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5",
         "mcp_servers": [{"name": "memory", "status": "connected"}]},
        {"type": "assistant", "message": {"model": "claude-opus-5-5", "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "t1", "name": "mcp__memory__search_memory",
             "input": {"query": question}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": [{"type": "text", "text": json.dumps({"as_of": "x", "results": []})}]}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t2", "name": "mcp__memory__get_latest_context",
             "input": {"hours": 48}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": True}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "draft"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "duration_ms": 1234,
         "num_turns": 3, "result": "Answer to: " + question},
    ]
    print("not json, ignored")
    for ev in events:
        print(json.dumps(ev))
    sys.exit(int(os.environ.get("FAKE_CLAUDE_EXIT", "0")))
    """
)


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "fake_claude.py"
    script.write_text(FAKE_CLAUDE, encoding="utf-8")
    exe = bin_dir / "claude"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    return log


@pytest.fixture
def setup(tmp_path):
    db = tmp_path / "c.db"
    db.write_bytes(b"")
    questions = tmp_path / "q.yaml"
    questions.write_text(
        "questions:\n"
        "  - id: C1\n    scenario: c\n    question: What time is dinner with Sam tomorrow?\n"
        "    required: [x]\n"
        "  - id: C7\n    scenario: c\n    question: What's Sam's flight number?\n"
        "    required: [y]\n",
        encoding="utf-8",
    )
    return db, questions, tmp_path / "out"


NOW = "2026-11-26T20:00:00+05:30"


def run_main(db, questions, out, *extra):
    rq.main(
        [
            "--system", "baseline", "--db", str(db), "--now", NOW, "--questions", str(questions),
            "--runs", "2", "--out", str(out), *extra,
        ]
    )  # fmt: skip


def calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


@pytest.mark.skipif(os.name == "nt", reason="fake claude is a POSIX shell script")
def test_runs_every_question_and_saves_results(fake_claude, setup):
    db, questions, out = setup
    run_main(db, questions, out)
    assert sorted(p.name for p in out.iterdir()) == [
        "C1-1.json",
        "C1-2.json",
        "C7-1.json",
        "C7-2.json",
    ]

    seen = calls(fake_claude)
    assert len(seen) == 4
    assert len({c["cwd"] for c in seen}) == 4
    for c in seen:
        assert c["cwd_files"] == []
        assert c["argv"][:2] == ["-p", c["argv"][1]]
        assert c["argv"][2:] == [
            "--model", "claude-opus-5-5",
            "--strict-mcp-config",
            "--mcp-config", c["argv"][6],
            "--allowedTools", "mcp__memory",
            "--tools", "",
            "--setting-sources", "",
            "--system-prompt",
            "You answer questions about the user's life using only the memory tools. The as_of"
            " time the tools return is the current time; ignore any other clock. Give full"
            " dates. Quote the message that supports each claim. If memory has no answer, say"
            " you don't know.",
            "--output-format", "stream-json",
            "--verbose",
        ]  # fmt: skip
        assert not Path(c["argv"][6]).is_relative_to(c["cwd"])
        assert c["config"] == {
            "mcpServers": {
                "memory": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [
                        "-W", "ignore", "-m", "baseline.server",
                        "--db", str(db.resolve()), "--now", NOW,
                    ],
                    "cwd": str(rq.REPO_ROOT),
                }
            }
        }  # fmt: skip
    assert [c["argv"][1] for c in seen] == [
        "What time is dinner with Sam tomorrow?",
        "What time is dinner with Sam tomorrow?",
        "What's Sam's flight number?",
        "What's Sam's flight number?",
    ]

    data = json.loads((out / "C7-2.json").read_text(encoding="utf-8"))
    assert data["id"] == "C7"
    assert data["question"] == "What's Sam's flight number?"
    assert data["run"] == 2
    assert data["answer"] == "Answer to: What's Sam's flight number?"
    assert data["model"] == "claude-opus-5-5"
    assert data["exit_code"] == 0
    assert data["duration_ms"] == 1234
    assert isinstance(data["duration_seconds"], float)
    assert data["mcp_servers"] == [{"name": "memory", "status": "connected"}]
    assert data["tool_calls"] == [
        {
            "id": "t1",
            "name": "mcp__memory__search_memory",
            "input": {"query": "What's Sam's flight number?"},
            "result": {"as_of": "x", "results": []},
            "is_error": False,
        },
        {
            "id": "t2",
            "name": "mcp__memory__get_latest_context",
            "input": {"hours": 48},
            "result": "boom",
            "is_error": True,
        },
    ]


@pytest.mark.skipif(os.name == "nt", reason="fake claude is a POSIX shell script")
def test_skips_existing_unless_forced(fake_claude, setup, capsys):
    db, questions, out = setup
    out.mkdir()
    (out / "C1-1.json").write_text("{}", encoding="utf-8")
    run_main(db, questions, out, "--only", "C1")
    assert len(calls(fake_claude)) == 1
    assert (out / "C1-1.json").read_text(encoding="utf-8") == "{}"
    assert "C1-1.json: exists, skipping" in capsys.readouterr().out

    run_main(db, questions, out, "--only", "C1", "--force")
    assert len(calls(fake_claude)) == 3
    assert json.loads((out / "C1-1.json").read_text(encoding="utf-8"))["run"] == 1


@pytest.mark.skipif(os.name == "nt", reason="fake claude is a POSIX shell script")
def test_nonzero_exit_is_saved_and_reported(fake_claude, setup, monkeypatch):
    db, questions, out = setup
    monkeypatch.setenv("FAKE_CLAUDE_EXIT", "3")
    with pytest.raises(SystemExit) as exc:
        run_main(db, questions, out, "--only", "C7")
    assert exc.value.code == 1
    assert json.loads((out / "C7-1.json").read_text(encoding="utf-8"))["exit_code"] == 3


def test_unknown_system_and_missing_db(setup):
    db, questions, out = setup
    with pytest.raises(SystemExit):
        rq.main(["--system", "engine", "--db", str(db), "--now", NOW,
                 "--questions", str(questions), "--out", str(out)])  # fmt: skip
    with pytest.raises(SystemExit):
        rq.main(["--system", "baseline", "--db", str(db) + "x", "--now", NOW,
                 "--questions", str(questions), "--out", str(out)])  # fmt: skip


def test_module_override_for_another_system():
    cfg = rq.mcp_config("sme.engine.server", "c.db", NOW)["mcpServers"]["memory"]
    assert cfg["args"][:4] == ["-W", "ignore", "-m", "sme.engine.server"]
    assert cfg["env"] == {"HF_HUB_OFFLINE": "1"}


NPM_SHIM = r"""@ECHO off
GOTO start
:find_dp0
SET dp0=%~dp0
EXIT /b
:start
SETLOCAL
CALL :find_dp0

IF EXIST "%dp0%\node.exe" (
  SET "_prog=%dp0%\node.exe"
) ELSE (
  SET "_prog=node"
  SET PATHEXT=%PATHEXT:;.JS;=;%
)

endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  """ + (
    r'"%dp0%\node_modules\@anthropic-ai\claude-code\cli.js" %*' + "\n"
)


def make_shim(tmp_path, with_node=True, with_script=True):
    shim = tmp_path / "claude.cmd"
    shim.write_text(NPM_SHIM, encoding="utf-8")
    script = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
    if with_script:
        script.parent.mkdir(parents=True)
        script.write_text("", encoding="utf-8")
    if with_node:
        (tmp_path / "node.exe").write_bytes(b"")
    return shim, script


def test_windows_cmd_shim_runs_node_directly(tmp_path):
    shim, script = make_shim(tmp_path)
    assert rq.claude_command(str(shim), windows=True) == [
        str(tmp_path / "node.exe"),
        str(script),
    ]


def test_windows_cmd_shim_uses_node_on_path(tmp_path, monkeypatch):
    shim, script = make_shim(tmp_path, with_node=False)
    monkeypatch.setattr(rq.shutil, "which", lambda name: "/opt/node/node.exe")
    assert rq.claude_command(str(shim), windows=True) == ["/opt/node/node.exe", str(script)]


def test_windows_cmd_shim_fallback_checks_arguments(tmp_path, monkeypatch):
    shim, _ = make_shim(tmp_path, with_script=False)
    prefix = rq.claude_command(str(shim), windows=True)
    assert prefix == [str(shim)]
    argv = rq.claude_argv(prefix, "What's on this weekend?", tmp_path / "mcp.json")
    assert argv[0] == str(shim) and argv[argv.index("--tools") + 1] == ""
    with pytest.raises(ValueError, match="cmd shim"):
        rq.claude_argv(prefix, 'Say "100%" & more', tmp_path / "mcp.json")


NATIVE_SHIM = (
    "@ECHO off\r\nSETLOCAL\r\nCALL :find_dp0\r\n"
    + r'"%dp0%\node_modules\@anthropic-ai\claude-code\bin\claude.exe"   %*'
    + "\r\n"
)


def test_windows_cmd_shim_to_native_exe_runs_exe_directly(tmp_path):
    shim = tmp_path / "claude.cmd"
    shim.write_text(NATIVE_SHIM, encoding="utf-8")
    exe = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert rq.claude_command(str(shim), windows=True) == [str(exe)]


def test_plain_executable(tmp_path):
    assert rq.claude_command("/usr/bin/claude", windows=False) == ["/usr/bin/claude"]
    assert rq.claude_command("C:\\bin\\claude.exe", windows=True) == ["C:\\bin\\claude.exe"]
    with pytest.raises(FileNotFoundError):
        rq.claude_command(None)


def test_parse_stream_without_result_event():
    lines = [
        json.dumps({"type": "assistant", "message": {"model": "m", "content": [
            {"type": "text", "text": "partial answer"}]}}),
    ]  # fmt: skip
    parsed = rq.parse_stream("\n".join(lines))
    assert parsed["answer"] == "partial answer"
    assert parsed["model"] == "m"
    assert parsed["tool_calls"] == []
