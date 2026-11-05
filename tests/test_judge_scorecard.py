from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from runner import judge as jd
from runner import scorecard as sc

QUESTION = {
    "id": "C1",
    "question": "What time is dinner with Sam tomorrow?",
    "required": ["9:30 PM", "Friday November 27", "never says the dinner is at 8"],
}
NOW = "2026-11-26T20:00:00+05:30 (Thursday). Asia/Kolkata (IST)."


def result(qid="C1", run=1, answer="Dinner is Friday November 27 at 9:30 PM.", **kw):
    data = {
        "id": qid,
        "question": QUESTION["question"],
        "run": run,
        "answer": answer,
        "tool_calls": [],
        "model": "claude-opus-5-5",
        "duration_seconds": 10.0,
        "exit_code": 0,
        "is_error": False,
        "num_turns": 3,
        "mcp_servers": [],
    }
    data.update(kw)
    return data


def reply(*mets: bool, prose: str = "") -> str:
    elements = [{"index": i, "met": m, "why": f"reason {i}"} for i, m in enumerate(mets, 1)]
    return prose + json.dumps({"elements": elements}) + prose


def write(folder: Path, name: str, data: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- judge: parsing


def test_outermost_json_ignores_prose_and_fences():
    text = 'Here you go:\n```json\n{"elements": [{"index": 1, "met": true, "why": "a {b}"}]}\n```'
    assert jd.outermost_json(text) == {"elements": [{"index": 1, "met": True, "why": "a {b}"}]}


def test_outermost_json_falls_back_when_trailing_braces_break_the_span():
    text = '{"elements": []} and then a stray } brace'
    assert jd.outermost_json(text) == {"elements": []}


def test_outermost_json_raises_without_object():
    with pytest.raises(ValueError):
        jd.outermost_json("no json here")


def test_parse_verdict_maps_by_index_and_keeps_required_text():
    text = json.dumps(
        {
            "elements": [
                {"index": 2, "met": False, "why": "no date"},
                {"index": 1, "met": True, "why": "says 9:30 PM"},
                {"index": 3, "met": True, "why": "8 only as history"},
            ]
        }
    )
    out = jd.parse_verdict(text, QUESTION["required"])
    assert [e["required"] for e in out] == QUESTION["required"]
    assert [e["met"] for e in out] == [True, False, True]
    assert out[1]["why"] == "no date"


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"elements": [{"index": 1, "met": True}]}),
        json.dumps({"elements": [{"met": "yes"}, {"met": True}, {"met": True}]}),
        json.dumps({"verdict": "pass"}),
    ],
)
def test_parse_verdict_rejects_wrong_shape(text):
    with pytest.raises(ValueError):
        jd.parse_verdict(text, QUESTION["required"])


# --- judge: verdicts


def test_judge_result_pass_and_fail():
    ok = jd.judge_result(result(), QUESTION, NOW, lambda p: reply(True, True, True))
    assert ok["pass"] is True and ok["judge_model"] == jd.JUDGE_MODEL
    assert ok["id"] == "C1" and ok["run"] == 1
    bad = jd.judge_result(result(), QUESTION, NOW, lambda p: reply(True, False, True))
    assert bad["pass"] is False


def test_prompt_has_answer_and_elements_but_no_tool_calls():
    seen = []
    call = {"id": "t1", "name": "x", "input": {}, "result": "SECRET_TOOL_OUTPUT"}
    res = result(tool_calls=[call])
    jd.judge_result(res, QUESTION, NOW, lambda p: seen.append(p) or reply(True, True, True))
    prompt = seen[0]
    assert "SECRET_TOOL_OUTPUT" not in prompt
    assert res["answer"] in prompt and NOW in prompt and QUESTION["question"] in prompt
    for r in QUESTION["required"]:
        assert r in prompt


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({"answer": ""}, "empty answer"),
        ({"answer": None}, "empty answer"),
        ({"answer": "   \n"}, "empty answer"),
        ({"exit_code": 1}, "exit code 1"),
        ({"exit_code": None}, "exit code None"),
    ],
)
def test_failed_runs_fail_without_judge_call(kw, reason):
    def boom(prompt):
        raise AssertionError("judge must not be called")

    v = jd.judge_result(result(**kw), QUESTION, NOW, boom)
    assert v["pass"] is False and v["judge_model"] is None
    assert len(v["elements"]) == 3
    assert all(not e["met"] and reason in e["why"] for e in v["elements"])


def test_judge_result_retries_then_raises():
    calls = []

    def flaky(prompt):
        calls.append(1)
        return "not json" if len(calls) == 1 else reply(True, True, True)

    assert jd.judge_result(result(), QUESTION, NOW, flaky)["pass"] is True
    with pytest.raises(RuntimeError):
        jd.judge_result(result(), QUESTION, NOW, lambda p: "garbage")


def test_judge_folder_skips_existing_verdicts_and_verdict_files(tmp_path):
    write(tmp_path, "C1-1.json", result(run=1))
    write(tmp_path, "C1-2.json", result(run=2, answer=""))
    write(tmp_path, "C1-3.json", result(run=3))
    write(tmp_path, "C1-3.verdict.json", {"id": "C1", "run": 3, "pass": True})
    write(tmp_path, "C9-1.json", result(qid="C9"))
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return reply(True, True, True)

    judged, errors = jd.judge_folder(tmp_path, [QUESTION], NOW, fake, log=lambda s: None)
    assert (judged, errors) == (2, 0)
    assert len(prompts) == 1
    v1 = json.loads((tmp_path / "C1-1.verdict.json").read_text(encoding="utf-8"))
    assert v1["pass"] is True and set(v1) == {"id", "run", "elements", "pass", "judge_model"}
    v2 = json.loads((tmp_path / "C1-2.verdict.json").read_text(encoding="utf-8"))
    assert v2["pass"] is False
    assert not (tmp_path / "C1-1.verdict.verdict.json").exists()

    judged, _ = jd.judge_folder(tmp_path, [QUESTION], NOW, fake, force=True, log=lambda s: None)
    assert judged == 3


def test_judge_folder_counts_errors_and_writes_nothing_for_them(tmp_path):
    write(tmp_path, "C1-1.json", result())
    judged, errors = jd.judge_folder(tmp_path, [QUESTION], NOW, lambda p: "x", log=lambda s: None)
    assert (judged, errors) == (0, 1)
    assert not (tmp_path / "C1-1.verdict.json").exists()


def test_scenario_now_reads_yaml_header():
    now = jd.scenario_now(Path(__file__).resolve().parents[1] / "corpus/scenario_c_questions.yaml")
    assert "2026-11-26T20:00:00+05:30" in now and "Thursday" in now and "IST" in now


def test_claude_judge_argv_and_envelope(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["input"] = argv, kw["input"]
        cfg = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text(encoding="utf-8"))
        seen["config"] = cfg
        out = json.dumps({"type": "result", "is_error": False, "result": reply(True)})
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(jd.subprocess, "run", fake_run)
    text = jd.claude_judge(["claude"])("PROMPT")
    assert jd.outermost_json(text)["elements"][0]["met"] is True
    argv = seen["argv"]
    assert seen["input"] == "PROMPT" and "PROMPT" not in argv
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert seen["config"] == {"mcpServers": {}}
    assert argv[argv.index("--system-prompt") + 1] == jd.SYSTEM_PROMPT


def test_claude_judge_nonzero_exit_is_a_value_error(monkeypatch):
    monkeypatch.setattr(
        jd.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 2, stdout="", stderr="bad"),
    )
    with pytest.raises(ValueError):
        jd.claude_judge(["claude"])("PROMPT")


# --- scorecard


def test_memory_tokens_counts_strings_raw_and_json_dumped_otherwise():
    calls = [
        {"result": "a" * 40},
        {"result": {"k": "v"}},
        {"result": None},
    ]
    chars = 40 + len(json.dumps({"k": "v"})) + len("null")
    assert sc.memory_tokens({"tool_calls": calls}) == round(chars / 4)
    assert sc.memory_tokens({"tool_calls": []}) == 0


def make_system(folder: Path, spec: dict[str, list[bool | None]], calls: int = 2) -> None:
    for qid, passes in spec.items():
        for run, p in enumerate(passes, 1):
            tool_calls = [{"result": "x" * 400} for _ in range(calls)]
            write(folder, f"{qid}-{run}.json", result(qid=qid, run=run, tool_calls=tool_calls))
            if p is not None:
                write(folder, f"{qid}-{run}.verdict.json", {"id": qid, "run": run, "pass": p})


def test_scorecard_aggregation(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_system(a, {"C1": [True, True, True], "C2": [True, False, None], "C10": [True] * 3})
    make_system(b, {"C1": [False, False, False]}, calls=4)
    card = sc.build([("a", str(a)), ("b", str(b)), ("c", str(tmp_path / "missing"))])

    assert card["question_ids"] == ["C1", "C2", "C10"]
    qa = card["systems"]["a"]["questions"]
    assert qa["C2"] == {
        "runs": 3,
        "judged": 2,
        "passed": 1,
        "mean_tool_calls": 2.0,
        "mean_memory_tokens": 200,
        "mean_duration_seconds": 10.0,
    }
    ta = card["systems"]["a"]["totals"]
    assert ta["questions_all_passed"] == 2
    assert (ta["passed"], ta["judged"], ta["unjudged"], ta["runs"]) == (7, 8, 1, 9)
    tb = card["systems"]["b"]["totals"]
    assert (tb["questions_all_passed"], tb["passed"], tb["judged"]) == (0, 0, 3)
    assert tb["mean_tool_calls"] == 4.0 and tb["mean_memory_tokens"] == 400
    assert card["systems"]["b"]["questions"]["C2"]["judged"] == 0
    assert card["systems"]["c"]["totals"]["runs"] == 0


def test_scorecard_markdown(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_system(a, {"C1": [True, True, True], "C2": [True, False, None]})
    make_system(b, {"C1": [None, None, None]})
    md = sc.render(sc.build([("a", str(a)), ("b", str(b))]))
    lines = md.splitlines()
    assert sc.TOKENS_LABEL in lines[0]
    assert "| C1 | a | 3/3 | 2.0 | 200 | 10.0 |" in lines
    assert "| C2 | a | 1/2 | 2.0 | 200 | 10.0 |" in lines
    assert "| C1 | b | - | 2.0 | 200 | 10.0 |" in lines
    assert "| C2 | b | - | - | - | - |" in lines
    assert "| a | 1/2 | 4/5 | 2.0 | 200 | 10.0 |" in lines
    assert "| b | - | - | 2.0 | 200 | 10.0 |" in lines
    assert any("a: 1 of 6 runs have no verdict" in line for line in lines)
    assert any("b: 3 of 3 runs have no verdict" in line for line in lines)


def test_scorecard_main_writes_md_and_json(tmp_path, capsys):
    a = tmp_path / "a"
    make_system(a, {"C1": [True, True, True]})
    out = tmp_path / "out" / "scorecard.md"
    sc.main(["--system", f"a={a}", "--system", f"h={tmp_path / 'none'}", "--out", str(out)])
    assert out.read_text(encoding="utf-8") == capsys.readouterr().out
    card = json.loads((out.parent / "scorecard.json").read_text(encoding="utf-8"))
    assert card["systems"]["a"]["totals"]["questions_all_passed"] == 1
    assert card["systems"]["h"]["totals"]["runs"] == 0
