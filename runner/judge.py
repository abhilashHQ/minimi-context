"""Grade saved answers against each question's required elements with an LLM judge.

    python -m runner.judge --results runs/score_d/minimi_shape \\
        --questions corpus/scenario_c_questions.yaml [--force]

For every `<id>-<run>.json` result without a `<id>-<run>.verdict.json` beside it
(or every result with --force), `claude -p` grades the final answer element by
element and the verdict is written next to the result. The judge sees only the
question, the required elements, the scenario "now" and the answer text, never
the tool calls. A run that exited non-zero or gave no answer fails every element
without a judge call. A human audits the verdicts.
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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from runner.run_questions import CMD_UNSAFE, claude_command, load_questions, write_result

JUDGE_MODEL = "claude-opus-5-5"
WORKERS = 4
TIMEOUT_SECONDS = 600
ATTEMPTS = 2
DEFAULT_NOW = "2026-11-26 20:00 IST (Asia/Kolkata), Thursday"
RESULT_NAME = re.compile(r"^(?P<id>[A-Za-z0-9_]+)-(?P<run>\d+)\.json$")
NOW_LINE = re.compile(r'^#\s*Scenario \w+ "now":\s*(.+)$', re.MULTILINE | re.IGNORECASE)
NOW_NOTE = re.compile(r"^#\s*All times are (.+?)\.?\s*$", re.MULTILINE)

SYSTEM_PROMPT = (
    "You are a strict grader. You check one answer from a personal memory assistant"
    " against a list of required elements and decide, for each element separately,"
    " whether the answer meets it. You judge only the answer text you are given. You do"
    " not know the user's data beyond what the question and the elements say, and you"
    " never fill gaps with guesses in the answer's favour. You reply with one JSON object"
    " and nothing else."
)

RULES = """\
Rules:
- A positive element is met only if the answer clearly and fully states it. Equivalent
  wording and formats are fine (9:30 PM = 21:30; "tomorrow" alone does not give a date,
  but "tomorrow, Friday November 27" does). Missing, vague or partly wrong means not met.
- An element phrased as "never says X", "does not list X" or "none of X listed as Y" is
  met unless the answer asserts X as current or true now. Mentioning X as past history,
  as something that was changed or superseded, or as something already done is fine.
- Ids like c06 or c34 name source messages you cannot see. An element that says
  "quoting cNN" is met when the answer quotes, verbatim in quotation marks, a message
  that supports that element's specific claim. The id itself need not appear.
- When an element depends on facts you cannot see (for example "all 5 open commitments"),
  check what you can (the count, the due dates, internal consistency) and say in "why"
  what you could not verify.
- Elements are independent. Do not let one element's result change another's.
- Judge dates and days against the current time given above."""


def scenario_now(questions_path: str | Path) -> str:
    """The scenario's current time from the questions YAML header comments."""
    text = Path(questions_path).read_text(encoding="utf-8")
    match = NOW_LINE.search(text)
    if not match:
        return DEFAULT_NOW
    now = match.group(1).strip()
    note = NOW_NOTE.search(text)
    return f"{now}. {note.group(1)}." if note else now


def judge_prompt(question: dict[str, Any], answer: str, now: str) -> str:
    elements = "\n".join(f"{i}. {r}" for i, r in enumerate(question["required"], 1))
    n = len(question["required"])
    return f"""\
Current time in the scenario: {now}

Question the user asked:
{question["question"]}

Required elements ({n}):
{elements}

{RULES}

Answer to grade:
<answer>
{answer}
</answer>

Reply with only this JSON object, one entry per required element, in the same order:
{{"elements": [{{"index": 1, "met": true, "why": "one short sentence"}}, ...]}}
It must have exactly {n} entries."""


def outermost_json(text: str) -> Any:
    """The first complete top-level JSON object in a reply, ignoring prose and fences."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in judge reply")
    try:
        return json.loads(text[start : end + 1])
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no parseable JSON object in judge reply")


def parse_verdict(reply: str, required: list[str]) -> list[dict[str, Any]]:
    data = outermost_json(reply)
    items = data.get("elements") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != len(required):
        got = len(items) if isinstance(items, list) else None
        raise ValueError(f"judge returned {got} elements, expected {len(required)}")
    if all(isinstance(it, dict) and isinstance(it.get("index"), int) for it in items):
        items = sorted(items, key=lambda it: it["index"])
    out = []
    for req, item in zip(required, items, strict=True):
        if not isinstance(item, dict) or not isinstance(item.get("met"), bool):
            raise ValueError(f"bad element in judge reply: {item!r}")
        out.append({"required": req, "met": item["met"], "why": str(item.get("why", "")).strip()})
    return out


def failed_verdict(result: dict[str, Any], question: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": result["id"],
        "run": result["run"],
        "elements": [{"required": r, "met": False, "why": reason} for r in question["required"]],
        "pass": False,
        "judge_model": None,
    }


def run_failure(result: dict[str, Any]) -> str | None:
    if result.get("exit_code") != 0:
        return f"run failed: exit code {result.get('exit_code')}"
    if not (result.get("answer") or "").strip():
        return "run failed: empty answer"
    return None


JudgeFn = Callable[[str], str]


def judge_result(
    result: dict[str, Any], question: dict[str, Any], now: str, judge: JudgeFn
) -> dict[str, Any]:
    reason = run_failure(result)
    if reason:
        return failed_verdict(result, question, reason)
    prompt = judge_prompt(question, result["answer"], now)
    error: Exception | None = None
    for _ in range(ATTEMPTS):
        try:
            elements = parse_verdict(judge(prompt), question["required"])
        except ValueError as exc:
            error = exc
            continue
        return {
            "id": result["id"],
            "run": result["run"],
            "elements": elements,
            "pass": all(e["met"] for e in elements),
            "judge_model": JUDGE_MODEL,
        }
    raise RuntimeError(f"{result['id']}-{result['run']}: {error}")


def judge_argv(prefix: list[str], config_path: str | Path) -> list[str]:
    argv = [
        *prefix,
        "-p",
        "--model",
        JUDGE_MODEL,
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        str(config_path),
        "--setting-sources",
        "",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--output-format",
        "json",
    ]
    if prefix[0].lower().endswith((".cmd", ".bat")):
        bad = [a for a in argv[1:] if CMD_UNSAFE.search(a)]
        if bad:
            raise ValueError(f"cannot pass these through a .cmd shim safely: {bad}")
    return argv


def claude_judge(prefix: list[str], timeout: float = TIMEOUT_SECONDS) -> JudgeFn:
    """A judge that sends the prompt on stdin to `claude -p` with no tools and no MCP."""

    def judge(prompt: str) -> str:
        with (
            tempfile.TemporaryDirectory(prefix="sme-judge-cfg-") as cfg_dir,
            tempfile.TemporaryDirectory(prefix="sme-judge-") as work_dir,
        ):
            config_path = Path(cfg_dir) / "mcp.json"
            config_path.write_text(json.dumps({"mcpServers": {}}), "utf-8")
            proc = subprocess.run(
                judge_argv(prefix, config_path),
                cwd=work_dir,
                input=prompt,
                env=dict(os.environ),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        if proc.returncode != 0:
            raise ValueError(f"judge exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
        try:
            envelope = json.loads(proc.stdout)
        except ValueError:
            return proc.stdout
        if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
            if envelope.get("is_error"):
                raise ValueError(f"judge error: {envelope['result'][:500]}")
            return envelope["result"]
        return proc.stdout

    return judge


def verdict_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.stem + ".verdict.json")


def result_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.json") if RESULT_NAME.match(p.name))


def judge_folder(
    folder: str | Path,
    questions: list[dict[str, Any]],
    now: str,
    judge: JudgeFn,
    force: bool = False,
    workers: int = WORKERS,
    log: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Judge every result in `folder` that needs it. Returns (judged, errors)."""
    by_id = {q["id"]: q for q in questions}
    todo = []
    for path in result_files(Path(folder)):
        if verdict_path(path).exists() and not force:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        question = by_id.get(result.get("id"))
        if question is None:
            log(f"{path}: question {result.get('id')!r} not in the questions file, skipping")
            continue
        todo.append((path, result, question))

    def one(item: tuple[Path, dict[str, Any], dict[str, Any]]) -> tuple[Path, str]:
        path, result, question = item
        try:
            verdict = judge_result(result, question, now, judge)
        except Exception as exc:
            return path, f"{type(exc).__name__}: {exc}"
        write_result(verdict_path(path), verdict)
        met = sum(e["met"] for e in verdict["elements"])
        status = "PASS" if verdict["pass"] else "FAIL"
        return path, f"{status} {met}/{len(verdict['elements'])}"

    judged = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, status in pool.map(one, todo):
            if status.startswith(("PASS", "FAIL")):
                judged += 1
                log(f"{path.name}: {status}")
            else:
                errors += 1
                log(f"{path.name}: ERROR {status}")
    return judged, errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Judge saved answers with claude -p")
    parser.add_argument("--results", required=True, help="folder of <id>-<run>.json results")
    parser.add_argument("--questions", required=True, help="questions YAML")
    parser.add_argument("--force", action="store_true", help="rejudge results that have a verdict")
    parser.add_argument("--claude", help="path to the claude executable (default: PATH)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    if not Path(args.results).is_dir():
        parser.error(f"results folder not found: {args.results}")
    prefix = claude_command(args.claude or shutil.which("claude"))
    questions = load_questions(args.questions)
    now = scenario_now(args.questions)
    judged, errors = judge_folder(
        args.results,
        questions,
        now,
        claude_judge(prefix, args.timeout),
        force=args.force,
        workers=args.workers,
        log=lambda line: print(line, flush=True),
    )
    print(f"{judged} judged, {errors} errors", flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
