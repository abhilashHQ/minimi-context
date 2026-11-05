from __future__ import annotations

import copy
import json

import pytest

from corpus import check, scenario_c


@pytest.fixture
def planted():
    return scenario_c.read_json(scenario_c.PLANTED_PATH)


@pytest.fixture
def corpus(scenario_c_corpus):
    return copy.deepcopy(scenario_c_corpus)


def noise_in(corpus, thread, sender=None):
    return next(
        m
        for m in corpus["messages"]
        if m["label"].startswith("n")
        and m["thread"] == thread
        and (sender is None or m["sender"] == sender)
    )


def failures_for(corpus, planted, label):
    return [f for f in check.check(corpus, planted) if f.startswith(f"{label}:")]


def test_clean_corpus_passes(corpus, planted):
    assert check.check(corpus, planted) == []


def test_main_exit_codes(tmp_path, corpus, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(corpus), encoding="utf-8")
    check.main([str(good)])
    assert capsys.readouterr().out.startswith("OK:")

    msg = noise_in(corpus, "college")
    msg["body"] = "Saw the Diwali photos"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        check.main([str(bad)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert f"{msg['label']}: banned term 'Diwali'" in out
    assert f"{msg['label']}: banned term 'photo'" in out
    assert "FAIL: 2 failures" in out


def test_banned_word(corpus, planted):
    msg = noise_in(corpus, "sam", "Sam")
    msg["body"] = "That DINNERS thing was wild"
    assert failures_for(corpus, planted, msg["label"]) == [
        f"{msg['label']}: banned term 'dinner': 'That DINNERS thing was wild'"
    ]


def test_you_promise_in_noise(corpus, planted):
    msg = noise_in(corpus, "flat-4b", "You")
    msg["body"] = "Fine, I’ll sort the shelf"
    found = failures_for(corpus, planted, msg["label"])
    assert len(found) == 1
    assert 'You message has banned term "I\'ll"' in found[0]


def test_you_phrase_allowed_for_others(corpus, planted):
    msg = noise_in(corpus, "flat-4b", "Rohan")
    msg["body"] = "I'll be late, see you"
    assert failures_for(corpus, planted, msg["label"]) == []


def test_out_of_order(corpus, planted):
    msgs = corpus["messages"]
    i = next(i for i, m in enumerate(msgs) if m["thread"] == "family" and m["label"][0] == "n")
    msgs[i], msgs[i + 1] = msgs[i + 1], msgs[i]
    found = failures_for(corpus, planted, msgs[i + 1]["label"])
    assert found and "out of time order" in found[0]


def test_wrong_planted_text(corpus, planted):
    msg = next(m for m in corpus["messages"] if m["label"] == "c25")
    msg["body"] = "Yes, same as last time."
    assert failures_for(corpus, planted, "c25") == [
        "c25: body is 'Yes, same as last time.', expected 'Yes, same as last time'"
    ]


def test_wrong_planted_sender_and_time(corpus, planted):
    msg = next(m for m in corpus["messages"] if m["label"] == "c01")
    msg["sender"] = "Sam"
    msg["sent_at"] = "2026-11-18T10:16:00+05:30"
    found = failures_for(corpus, planted, "c01")
    assert any("sender is 'Sam', expected 'Samuel Rao'" in f for f in found)
    assert any("sent_at is '2026-11-18T10:16:00+05:30'" in f for f in found)


def test_planted_missing_and_duplicated(corpus, planted):
    msgs = corpus["messages"]
    c13 = next(i for i, m in enumerate(msgs) if m["label"] == "c13")
    msgs.insert(c13, copy.deepcopy(msgs[c13]))
    msgs[:] = [m for m in msgs if m["label"] != "c40"]
    fails = check.check(corpus, planted)
    assert "c13: planted message appears 2 times" in fails
    assert "c40: planted message is missing" in fails


def test_out_of_range_time(corpus, planted):
    msgs = corpus["messages"]
    last = max(i for i, m in enumerate(msgs) if m["thread"] == "college")
    msgs[last]["sent_at"] = "2026-11-26T20:01:00+05:30"
    first = min(i for i, m in enumerate(msgs) if m["thread"] == "college")
    msgs[first]["sent_at"] = "2026-11-01T23:59:00+05:30"
    for i in (first, last):
        found = failures_for(corpus, planted, msgs[i]["label"])
        assert (
            found
            and "is outside 2026-11-02T00:00:00+05:30 to 2026-11-26T20:00:00+05:30" in (found[0])
        )


def test_boundaries_are_inside(corpus, planted):
    msgs = corpus["messages"]
    msgs[max(i for i, m in enumerate(msgs) if m["thread"] == "college")]["sent_at"] = (
        "2026-11-26T20:00:00+05:30"
    )
    assert check.check(corpus, planted) == []


def test_sender_outside_conversation(corpus, planted):
    msg = noise_in(corpus, "college", "Kabir")
    msg["sender"] = "Rohan"
    assert failures_for(corpus, planted, msg["label"]) == [
        f"{msg['label']}: sender 'Rohan' is not a member (You, Kabir, Zoya): {msg['body']!r}"
    ]


def test_target_miss(corpus, planted):
    sam = [m for m in corpus["messages"] if m["thread"] == "sam" and m["label"][0] == "n"]
    drop = {m["label"] for m in sam[:31]}
    corpus["messages"] = [m for m in corpus["messages"] if m["label"] not in drop]
    assert check.check(corpus, planted) == [
        "-: sam has 269 noise messages, target 300 allows 270 to 330"
    ]


def test_target_zero_means_none(corpus, planted):
    extra = copy.deepcopy(next(m for m in corpus["messages"] if m["label"] == "c09"))
    extra.update(label="n9999", sender="Dev Mehta", body="Numbers look tidy")
    msgs = corpus["messages"]
    msgs.insert(next(i for i, m in enumerate(msgs) if m["label"] == "c10"), extra)
    assert check.check(corpus, planted) == [
        "-: q4-numbers has 1 noise messages, target 0 allows 0 to 0"
    ]


def test_unknown_label(corpus, planted):
    noise_in(corpus, "sam")["label"] = "x1"
    assert "x1: label is neither a planted id nor a noise label" in check.check(corpus, planted)


def test_target_range():
    assert check.target_range(18) == (17, 19)
    assert check.target_range(300) == (270, 330)
    assert check.target_range(10) == (9, 11)
    assert check.target_range(0) == (0, 0)


@pytest.mark.parametrize(
    ("body", "hits"),
    [
        ("dinner", ["dinner"]),
        ("Two Dinners", ["dinner"]),
        ("the CAP  TABLE is fine", ["cap table", "table"]),
        ("timetable and platform and vegetables", []),
        ("a veg thali", ["veg"]),
        ("ticket, tickets", ["ticket"]),
        ("6E is a code", ["6E"]),
        ("11,200 and 1,2000", []),
        ("owes 1,200.", ["owe", "1,200"]),
        ("q4s done", ["Q4"]),
        ("Kapoor's notice", ["Kapoor"]),
        ("taps and tapestry", ["tap"]),
        ("forms, format", ["form"]),
        ("PAY back later", ["pay back"]),
        ("trains, training", ["train"]),
        ("sundays are slow", ["Sunday"]),
    ],
)
def test_banned_terms(body, hits):
    assert check.banned_hits(body) == hits


@pytest.mark.parametrize(
    ("body", "hits"),
    [
        ("i'll check", ["I'll"]),
        ("I will, I can", ["I will", "I can"]),
        ("Will do", ["will do"]),
        ("promises", ["promise"]),
        ("done by tomorrow", ["by tomorrow"]),
        ("by monday or by WEDNESDAY", ["by Monday", "by Wednesday"]),
        ("By Sunday", ["by Sunday"]),
        ("see you", ["see you"]),
        ("ok, nice one", []),
        ("Ill be fine", []),
    ],
)
def test_you_banned_terms(body, hits):
    assert check.you_banned_hits(body) == hits


def test_noise_rules_skip_sender_name():
    assert (
        check.noise_rule_violations("Mr. Kapoor", "Rent received, thanks", ("You", "Mr. Kapoor"))
        == []
    )
