from __future__ import annotations

import copy
import json

import pytest
import yaml

from corpus import scenario_c
from sme.store import repo

from .scenario_c_fixture import conversation_rows, parse_sent, planted_rows, question_rows

APPS = {"Gmail": "gmail", "WhatsApp": "whatsapp_web"}


@pytest.fixture
def planted():
    return scenario_c.read_json(scenario_c.PLANTED_PATH)


def test_planted_header(planted):
    assert planted["scenario"] == "c"
    assert planted["now"] == "2026-11-26T20:00:00+05:30"
    assert planted["conversations"] == [
        {"app": APPS[r["app"]], "key": r["key"], "title": r["title"]} for r in conversation_rows()
    ]


def test_planted_matches_spec_table(planted):
    rows = planted_rows()
    assert [r["id"] for r in rows] == [f"c{i:02d}" for i in range(1, 49)]
    app_of = {r["key"]: APPS[r["app"]] for r in conversation_rows()}
    by_label = {m["label"]: m for m in planted["messages"]}
    assert len(by_label) == len(planted["messages"]) == 48
    for r in rows:
        msg = by_label[r["id"]]
        app = app_of[r["thread"]]
        assert msg == {
            "label": r["id"],
            "app": app,
            "thread": r["thread"],
            "sender": r["from"],
            "sent_at": parse_sent(r["sent"]).strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "date_source": "absolute" if app == "gmail" else "separator",
            "body": r["text"],
        }, r["id"]


def test_planted_ingest_and_time_order(planted):
    order = [c["key"] for c in planted["conversations"]]
    keys = [(order.index(m["thread"]), repo.parse_time(m["sent_at"])) for m in planted["messages"]]
    assert keys == sorted(keys)


def test_planted_ranges_match_table(planted):
    for row in conversation_rows():
        labels = [m["label"] for m in planted["messages"] if m["thread"] == row["key"]]
        if row["planted"] == "none":
            assert labels == []
        else:
            first, last = row["planted"].split(" to ")
            assert labels == [f"c{i:02d}" for i in range(int(first[1:]), int(last[1:]) + 1)]


def test_conversation_table_matches_spec():
    rows = conversation_rows()
    assert [c.key for c in scenario_c.CONVERSATIONS] == [r["key"] for r in rows]
    for conv, row in zip(scenario_c.CONVERSATIONS, rows, strict=True):
        assert (conv.app, conv.title) == (APPS[row["app"]], row["title"])
        assert conv.noise_target == int(row["noise"].split(",")[0])
        assert conv.members[0] == "You"
    by_key = scenario_c.BY_KEY
    assert by_key["standup"].members == ("You", "Dev Mehta")
    assert by_key["standup"].weekday_time == "09:30"
    assert by_key["newsletter"].members == ("You", "The Weekly Byte")
    assert by_key["newsletter"].weekday_time == "07:00"
    assert by_key["college"].members == ("You", "Kabir", "Zoya")
    assert by_key["flat-4b"].members == ("You", "Rohan", "Meera")
    assert by_key["family"].members == ("You", "Mom", "Dad", "Anu")
    assert by_key["kapoor"].members == ("You", "Mr. Kapoor")


def test_questions_match_spec_table():
    data = yaml.safe_load(scenario_c.CORPUS_DIR.joinpath("scenario_c_questions.yaml").read_text())
    questions = data["questions"]
    rows = question_rows()
    assert [q["id"] for q in questions] == [r["id"] for r in rows] == [f"C{i}" for i in range(1, 9)]
    for q, r in zip(questions, rows, strict=True):
        assert set(q) == {"id", "scenario", "question", "required"}
        assert q["scenario"] == "c"
        assert q["question"] == r["question"]
        assert all(isinstance(x, str) for x in q["required"])
        assert "; ".join(q["required"]) == r["required"]


def test_merge_loads_through_repo(scenario_c_corpus, conn):
    corpus = scenario_c_corpus
    assert set(corpus) == {"scenario", "description", "now", "conversations", "messages"}
    result = repo.load_scenario(conn, copy.deepcopy(corpus))
    assert result.conversations_added == 12
    assert result.messages_added == len(corpus["messages"])
    assert set(result.message_ids) == {m["label"] for m in corpus["messages"]}
    convs = [(r["app"], r["external_key"], r["title"]) for r in repo.conversations(conn)]
    assert convs == [(c.app, c.key, c.title) for c in scenario_c.CONVERSATIONS]


def test_merge_order_and_labels(scenario_c_corpus):
    msgs = scenario_c_corpus["messages"]
    order = [c.key for c in scenario_c.CONVERSATIONS]
    keys = [(order.index(m["thread"]), repo.parse_time(m["sent_at"])) for m in msgs]
    assert keys == sorted(keys)
    noise = [m["label"] for m in msgs if m["label"].startswith("n")]
    assert noise == [f"n{i:04d}" for i in range(1, len(noise) + 1)]
    assert [m["label"] for m in msgs if m["label"].startswith("c")] == [
        f"c{i:02d}" for i in range(1, 49)
    ]
    for m in msgs:
        assert m["date_source"] == scenario_c.DATE_SOURCE[m["app"]]
        assert list(m) == ["label", "app", "thread", "sender", "sent_at", "date_source", "body"]


def test_build_command(tmp_path, scenario_c_corpus, capsys):
    from corpus import noise as noise_mod

    from .conftest import FakeNoiseLLM

    planted = scenario_c.read_json(scenario_c.PLANTED_PATH)
    noise_path = tmp_path / "noise.json"
    noise_mod.generate(FakeNoiseLLM(), noise_path, planted, log=lambda line: None)
    out = tmp_path / "scenario_c.json"
    with pytest.raises(SystemExit) as exc:
        scenario_c.main(["build", "--noise", str(noise_path), "--out", str(out)])
    assert exc.value.code == 0
    assert json.loads(out.read_text(encoding="utf-8")) == scenario_c_corpus
    assert repo.read_scenario(out)["scenario"] == "c"


def test_build_refuses_incomplete_noise(tmp_path, capsys):
    noise_path = tmp_path / "noise.json"
    scenario_c.write_json(noise_path, {"scenario": "c", "conversations": {"sam": {}}})
    with pytest.raises(SystemExit) as exc:
        scenario_c.main(["build", "--noise", str(noise_path), "--out", str(tmp_path / "o.json")])
    assert exc.value.code == 1
    assert "missing" in capsys.readouterr().out
    assert not (tmp_path / "o.json").exists()
