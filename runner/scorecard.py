"""Compare judged runs across systems in one markdown table.

    python -m runner.scorecard --system minimi_shape=runs/score_d/minimi_shape \\
        --system harness=runs/score_d/harness [--out runs/score_d/scorecard.md]

Per system and question: runs passed of runs judged, mean tool calls, mean
memory tokens read (characters of every tool result / 4) and mean duration.
The same numbers go to `scorecard.json` beside the markdown file. Runs without
a verdict show as "-" and count toward no pass total.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from runner.judge import result_files, verdict_path

RUNS_PER_QUESTION = 3
TOKENS_LABEL = "≈ tokens read from memory"


def memory_tokens(result: dict[str, Any]) -> int:
    chars = 0
    for call in result.get("tool_calls") or []:
        value = call.get("result")
        chars += len(value if isinstance(value, str) else json.dumps(value))
    return round(chars / 4)


def _mean(values: list[float], digits: int) -> float | int | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return round(mean) if digits == 0 else round(mean, digits)


def _qid_key(qid: str) -> tuple[str, int, str]:
    m = re.match(r"^([A-Za-z_]*)(\d+)(.*)$", qid)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (qid, 0, "")


def load_system(folder: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Question id -> one record per result file with its numbers and verdict."""
    runs: dict[str, list[dict[str, Any]]] = {}
    folder = Path(folder)
    if not folder.is_dir():
        return runs
    for path in result_files(folder):
        result = json.loads(path.read_text(encoding="utf-8"))
        vpath = verdict_path(path)
        verdict = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else None
        runs.setdefault(result["id"], []).append(
            {
                "run": result.get("run"),
                "tool_calls": len(result.get("tool_calls") or []),
                "memory_tokens": memory_tokens(result),
                "duration_seconds": result.get("duration_seconds"),
                "pass": None if verdict is None else bool(verdict.get("pass")),
            }
        )
    return runs


def summarize(
    runs: dict[str, list[dict[str, Any]]],
    question_ids: list[str],
    runs_per_question: int = RUNS_PER_QUESTION,
) -> dict[str, Any]:
    questions = {}
    for qid in question_ids:
        recs = runs.get(qid, [])
        judged = [r for r in recs if r["pass"] is not None]
        durations = [r["duration_seconds"] for r in recs if r["duration_seconds"] is not None]
        questions[qid] = {
            "runs": len(recs),
            "judged": len(judged),
            "passed": sum(r["pass"] for r in judged),
            "mean_tool_calls": _mean([r["tool_calls"] for r in recs], 1),
            "mean_memory_tokens": _mean([r["memory_tokens"] for r in recs], 0),
            "mean_duration_seconds": _mean(durations, 1),
        }
    all_recs = [r for qid in question_ids for r in runs.get(qid, [])]
    judged = [r for r in all_recs if r["pass"] is not None]
    durations = [r["duration_seconds"] for r in all_recs if r["duration_seconds"] is not None]
    totals = {
        "questions": len(question_ids),
        "questions_all_passed": sum(
            1 for q in questions.values() if q["passed"] == runs_per_question
        ),
        "runs": len(all_recs),
        "judged": len(judged),
        "passed": sum(r["pass"] for r in judged),
        "unjudged": len(all_recs) - len(judged),
        "mean_tool_calls": _mean([r["tool_calls"] for r in all_recs], 1),
        "mean_memory_tokens": _mean([r["memory_tokens"] for r in all_recs], 0),
        "mean_duration_seconds": _mean(durations, 1),
    }
    return {"questions": questions, "totals": totals}


def build(
    systems: list[tuple[str, str]], runs_per_question: int = RUNS_PER_QUESTION
) -> dict[str, Any]:
    loaded = {name: load_system(folder) for name, folder in systems}
    ids = sorted({qid for runs in loaded.values() for qid in runs}, key=_qid_key)
    return {
        "runs_per_question": runs_per_question,
        "question_ids": ids,
        "systems": {
            name: {"folder": str(folder), **summarize(loaded[name], ids, runs_per_question)}
            for name, folder in systems
        },
    }


def _num(value: float | None, digits: int) -> str:
    if value is None:
        return "-"
    return str(round(value)) if digits == 0 else f"{value:.{digits}f}"


def _pass_cell(passed: int, judged: int) -> str:
    return "-" if judged == 0 else f"{passed}/{judged}"


def render(card: dict[str, Any]) -> str:
    rpq = card["runs_per_question"]
    lines = [
        f"| Question | System | Passed / judged | Mean tool calls | Mean {TOKENS_LABEL}"
        " | Mean duration (s) |",
        "|---|---|---|---|---|---|",
    ]
    for qid in card["question_ids"]:
        for name, sys_card in card["systems"].items():
            q = sys_card["questions"][qid]
            lines.append(
                f"| {qid} | {name} | {_pass_cell(q['passed'], q['judged'])}"
                f" | {_num(q['mean_tool_calls'], 1)} | {_num(q['mean_memory_tokens'], 0)}"
                f" | {_num(q['mean_duration_seconds'], 1)} |"
            )
    lines += [
        "",
        f"| System | Questions with all {rpq} runs passed | Runs passed / judged"
        f" | Mean tool calls | Mean {TOKENS_LABEL} | Mean duration (s) |",
        "|---|---|---|---|---|---|",
    ]
    notes = []
    for name, sys_card in card["systems"].items():
        t = sys_card["totals"]
        all_passed = "-" if t["judged"] == 0 else f"{t['questions_all_passed']}/{t['questions']}"
        lines.append(
            f"| {name} | {all_passed} | {_pass_cell(t['passed'], t['judged'])}"
            f" | {_num(t['mean_tool_calls'], 1)} | {_num(t['mean_memory_tokens'], 0)}"
            f" | {_num(t['mean_duration_seconds'], 1)} |"
        )
        if t["runs"] == 0:
            notes.append(f"{name}: no result files in {sys_card['folder']}.")
        elif t["unjudged"]:
            notes.append(
                f'{name}: {t["unjudged"]} of {t["runs"]} runs have no verdict; they show as "-"'
                " and are left out of the pass totals (tool calls, tokens and duration"
                " still cover every run)."
            )
        missing = [qid for qid, q in sys_card["questions"].items() if q["runs"] < rpq]
        if t["runs"] and missing:
            notes.append(f"{name}: fewer than {rpq} result files for {', '.join(missing)}.")
    notes.append(
        f"{TOKENS_LABEL}: characters of every tool result in a run divided by 4, averaged"
        " over runs."
    )
    lines += ["", *(f"- {n}" for n in notes)]
    return "\n".join(lines) + "\n"


def parse_system(value: str) -> tuple[str, str]:
    name, sep, folder = value.partition("=")
    if not sep or not name or not folder:
        raise argparse.ArgumentTypeError(f"expected NAME=FOLDER, got {value!r}")
    return name, folder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scorecard of judged runs across systems")
    parser.add_argument(
        "--system", action="append", type=parse_system, required=True, help="NAME=FOLDER"
    )
    parser.add_argument("--out", default="runs/score_d/scorecard.md", help="markdown output")
    parser.add_argument("--runs", type=int, default=RUNS_PER_QUESTION, help="runs per question")
    args = parser.parse_args(argv)

    card = build(args.system, args.runs)
    text = render(card)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text, end="")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_name("scorecard.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
