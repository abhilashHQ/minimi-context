"""Run the gate questions headless through `claude -p` and save every answer.

    python -m runner.run_questions --system baseline --db runs/p1c/c.db \\
        --now 2026-11-26T20:00:00+05:30 --questions corpus/scenario_c_questions.yaml \\
        --runs 3 --out runs/p1b/baseline

For each question and run, `claude -p` starts in a fresh empty temp folder with
only the system under test as its MCP server, no built-in tools and no
settings files. The answer, every tool call with its result, the model,
duration and exit code go to `<out>/<question id>-<run>.json`. Runs whose file
already exists are skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5-5"
# The memory server loads its embedding model at startup, which can take longer
# than Claude Code's default MCP startup timeout.
MCP_STARTUP_TIMEOUT_MS = "120000"
SERVER_NAME = "memory"
SYSTEM_PROMPT = (
    "You answer questions about the user's life using only the memory tools. The as_of time"
    " the tools return is the current time; ignore any other clock. Give full dates. Quote"
    " the message that supports each claim. If memory has no answer, say you don't know."
)
# System under test -> MCP server module, started with --db and --now.
SYSTEMS = {
    "baseline": "baseline.server",
    "minimi_shape": "baseline.minimi_shape",
}
TIMEOUT_SECONDS = 900


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["questions"]


def mcp_config(module: str, db: str | Path, now: str) -> dict[str, Any]:
    return {
        "mcpServers": {
            SERVER_NAME: {
                "type": "stdio",
                "command": sys.executable,
                "args": [
                    "-W",
                    "ignore",
                    "-m",
                    module,
                    "--db",
                    str(Path(db).resolve()),
                    "--now",
                    now,
                ],
                "cwd": str(REPO_ROOT),
                # The embedding model is cached by the build; checking the Hub on
                # every start adds ~30 s and can hang past the startup timeout.
                "env": {"HF_HUB_OFFLINE": "1"},
            }
        }
    }


_CMD_TARGET = re.compile(r'"%~?dp0%?\\([^"]+?(\.js|\\claude\.exe))"', re.IGNORECASE)


def claude_command(found: str | None, windows: bool | None = None) -> list[str]:
    """argv prefix that starts Claude Code, given what shutil.which("claude") found.

    On Windows an npm install puts a `claude.cmd` shim on PATH. Running a .cmd
    sends the arguments through cmd.exe, which mangles quotes and empty
    strings, so the shim's target is run directly instead: the native
    `claude.exe` when the shim points at one, otherwise its node script via node.
    """
    if not found:
        raise FileNotFoundError("claude is not on PATH; install Claude Code or pass --claude")
    windows = os.name == "nt" if windows is None else windows
    path = Path(found)
    if not (windows and path.suffix.lower() in (".cmd", ".bat")):
        return [found]

    script = None
    try:
        match = _CMD_TARGET.search(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        match = None
    if match:
        target = path.parent / match.group(1).replace("\\", os.sep)
        if match.group(2).lower().endswith(".exe") and target.exists():
            return [str(target)]
        script = target
    native = path.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    if native.exists():
        return [str(native)]
    default = path.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
    if script is None or not script.exists():
        script = default if default.exists() else None
    node = path.parent / "node.exe"
    node_cmd = str(node) if node.exists() else shutil.which("node")
    if script is not None and node_cmd:
        return [node_cmd, str(script)]
    # Last resort: let Windows run the shim. Our arguments carry no cmd.exe
    # metacharacters, which claude_argv checks before this is used.
    return [str(path)]


CMD_UNSAFE = re.compile(r'[%^&|<>"!\r\n]')


def claude_argv(prefix: list[str], question: str, config_path: str | Path) -> list[str]:
    argv = [
        *prefix,
        "-p",
        question,
        "--model",
        MODEL,
        "--strict-mcp-config",
        "--mcp-config",
        str(config_path),
        "--allowedTools",
        "mcp__memory",
        "--tools",
        "",
        "--setting-sources",
        "",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    if prefix[0].lower().endswith((".cmd", ".bat")):
        bad = [a for a in argv[1:] if CMD_UNSAFE.search(a)]
        if bad:
            raise ValueError(f"cannot pass these through a .cmd shim safely: {bad}")
    return argv


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
        )
    return json.dumps(content)


def _result_value(content: Any) -> Any:
    """A tool result as JSON when the server returned JSON text, else as text."""
    text = _text(content)
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def parse_stream(stdout: str) -> dict[str, Any]:
    """Pull the answer, tool calls and model out of `--output-format stream-json`."""
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue

    model = None
    mcp_servers = None
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    texts: list[str] = []
    result: dict[str, Any] | None = None
    for ev in events:
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init":
            model = ev.get("model", model)
            mcp_servers = ev.get("mcp_servers")
        elif kind == "assistant":
            msg = ev.get("message", {})
            model = model or msg.get("model")
            for block in msg.get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    texts.append(block["text"])
                elif block.get("type") == "tool_use":
                    calls[block["id"]] = {
                        "id": block["id"],
                        "name": block.get("name"),
                        "input": block.get("input"),
                        "result": None,
                        "is_error": None,
                    }
                    order.append(block["id"])
        elif kind == "user":
            for block in ev.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    call = calls.get(block.get("tool_use_id"))
                    if call is not None:
                        call["result"] = _result_value(block.get("content"))
                        call["is_error"] = bool(block.get("is_error", False))
        elif kind == "result":
            result = ev

    answer = result.get("result") if result and isinstance(result.get("result"), str) else None
    if answer is None and texts:
        answer = texts[-1]
    return {
        "answer": answer,
        "tool_calls": [calls[i] for i in order],
        "model": model,
        "mcp_servers": mcp_servers,
        "result_event": result,
        "event_count": len(events),
    }


def run_one(
    prefix: list[str],
    question: dict[str, Any],
    run: int,
    module: str,
    db: str | Path,
    now: str,
    timeout: float = TIMEOUT_SECONDS,
) -> dict[str, Any]:
    with (
        tempfile.TemporaryDirectory(prefix="sme-cfg-") as cfg_dir,
        tempfile.TemporaryDirectory(prefix="sme-run-") as work_dir,
    ):
        config_path = Path(cfg_dir) / "mcp.json"
        config_path.write_text(json.dumps(mcp_config(module, db, now), indent=2), "utf-8")
        argv = claude_argv(prefix, question["question"], config_path)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                cwd=work_dir,
                stdin=subprocess.DEVNULL,
                env={**os.environ, "MCP_TIMEOUT": MCP_STARTUP_TIMEOUT_MS},
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\ntimed out after {timeout} seconds"
            stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
            exit_code = None
        duration = round(time.monotonic() - started, 3)

    parsed = parse_stream(stdout)
    result_event = parsed.pop("result_event") or {}
    return {
        "id": question["id"],
        "question": question["question"],
        "run": run,
        "answer": parsed["answer"],
        "tool_calls": parsed["tool_calls"],
        "model": parsed["model"],
        "duration_seconds": duration,
        "duration_ms": result_event.get("duration_ms"),
        "exit_code": exit_code,
        "is_error": result_event.get("is_error"),
        "num_turns": result_event.get("num_turns"),
        "mcp_servers": parsed["mcp_servers"],
        "server_module": module,
        "now": now,
        "argv": argv[len(prefix) :],
        "stderr": stderr,
        "event_count": parsed["event_count"],
    }


def write_result(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run gate questions through claude -p")
    parser.add_argument("--system", required=True, help=f"one of {', '.join(SYSTEMS)}")
    parser.add_argument("--module", help="server module to use instead of the system's default")
    parser.add_argument("--db", required=True, help="database the server reads")
    parser.add_argument("--now", required=True, help="ISO time with offset passed to the server")
    parser.add_argument("--questions", required=True, help="questions YAML")
    parser.add_argument("--runs", type=int, default=3, help="runs per question")
    parser.add_argument("--out", required=True, help="folder for <id>-<run>.json results")
    parser.add_argument("--only", help="comma separated question ids")
    parser.add_argument("--claude", help="path to the claude executable (default: PATH)")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument("--force", action="store_true", help="rerun runs that already have a file")
    args = parser.parse_args(argv)

    module = args.module or SYSTEMS.get(args.system)
    if module is None:
        parser.error(f"unknown system {args.system!r}; pass --module or use {sorted(SYSTEMS)}")
    if not Path(args.db).exists():
        parser.error(f"database not found: {args.db}")
    prefix = claude_command(args.claude or shutil.which("claude"))

    questions = load_questions(args.questions)
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    out = Path(args.out)
    failed = 0
    for q in questions:
        for run in range(1, args.runs + 1):
            path = out / f"{q['id']}-{run}.json"
            if path.exists() and not args.force:
                print(f"{path}: exists, skipping", flush=True)
                continue
            print(f"{q['id']} run {run}: {q['question']}", flush=True)
            data = run_one(prefix, q, run, module, args.db, args.now, args.timeout)
            write_result(path, data)
            calls = len(data["tool_calls"])
            print(
                f"  exit {data['exit_code']}, {calls} tool calls,"
                f" {data['duration_seconds']}s -> {path}",
                flush=True,
            )
            if data["exit_code"] != 0 or data["answer"] is None:
                failed += 1
    if failed:
        print(f"{failed} runs exited non-zero or gave no answer; see their stderr", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
