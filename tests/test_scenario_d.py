from __future__ import annotations

import copy
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import timedelta

import pytest

from corpus import scenario_d
from sme.store import repo

from .conftest import FakeEmbedder, FakeLLM

NOW = repo.parse_time(scenario_d.NOW)
BACKFILL = repo.parse_time(scenario_d.BACKFILL_AT)
C_COUNT = 1326
COPY_THREADS = {"friday-dinner", "investor-deck", "q4-numbers", "offsite", "standup", "newsletter"}
KNOWN_PEOPLE = re.compile(
    r"\b(Sam|Samuel|Priya|Arjun|Mom|Dad|Anu|Rohan|Meera|Kapoor|Kabir|Zoya|Mehta)\b"
)


@pytest.fixture(scope="module")
def d():
    return scenario_d.read_json(scenario_d.OUT_PATH)


@pytest.fixture(scope="module")
def c():
    return scenario_d.read_json(scenario_d.SOURCE_PATH)


def new_messages(d):
    return [m for m in d["messages"] if m["label"].startswith("d")]


def promos(d):
    return [m for m in d["messages"] if m["thread"].startswith("promo-")]


def store_conn(path):
    conn = repo.connect(path)
    conn.executescript(repo.SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def test_counts_by_app_and_source(d):
    counts = Counter((m["app"], m.get("source")) for m in d["messages"])
    assert counts == {
        ("whatsapp_web", "screen"): 1276,
        ("gmail", "screen"): 50,
        ("gmail_connector", "connector"): 50 + 300 + 4,
        ("calendar", "connector"): 8,
    }
    assert len(promos(d)) == 302
    assert len({m["external_id"] for m in promos(d)}) == 300


def test_scenario_c_kept_as_is(d, c):
    assert d["scenario"] == "d"
    assert d["now"] == c["now"] == scenario_d.NOW
    assert d["conversations"][: len(c["conversations"])] == c["conversations"]
    assert d["messages"][:C_COUNT] == [
        {**m, "source": "screen", "seen_at": m["sent_at"]} for m in c["messages"]
    ]
    for m in d["messages"]:
        assert m["body"].strip()


def test_order_and_labels(d):
    order = {(conv["app"], conv["key"]): i for i, conv in enumerate(d["conversations"])}
    assert len(order) == len(d["conversations"])
    keys = [(order[(m["app"], m["thread"])], repo.parse_time(m["sent_at"])) for m in d["messages"]]
    assert keys == sorted(keys)
    labels = [m["label"] for m in new_messages(d)]
    assert labels == [f"d{i:04d}" for i in range(1, len(labels) + 1)]
    assert [m["label"] for m in d["messages"][C_COUNT:]] == labels


def test_gmail_copies_match_screen_messages(d):
    screen = {
        (m["thread"], m["sender"], m["sent_at"]): m for m in d["messages"] if m["app"] == "gmail"
    }
    copies = [
        m for m in d["messages"] if m["app"] == "gmail_connector" and m["thread"] in COPY_THREADS
    ]
    assert len(copies) == 52
    assert {(m["thread"], m["sender"], m["sent_at"]) for m in copies} == set(screen)
    for m in copies:
        original = screen[(m["thread"], m["sender"], m["sent_at"])]
        assert m["body"].startswith(original["body"])
        extra = m["body"][len(original["body"]) :]
        assert ">" not in extra and "wrote:" not in extra
        assert m["labels"] == ["INBOX"]
        assert m["date_source"] == "absolute"
        assert m["sender_email"] == scenario_d.SENDER_EMAILS[m["sender"]]


def test_connector_sync_times(d):
    first: dict[str, dict] = {}
    for m in new_messages(d):
        first.setdefault(m["external_id"], m)
    for m in first.values():
        sent, seen = repo.parse_time(m["sent_at"]), repo.parse_time(m["seen_at"])
        if sent < BACKFILL:
            assert seen == BACKFILL, m["label"]
        else:
            assert timedelta(minutes=2) <= seen - sent <= timedelta(minutes=15), m["label"]


def test_duplicate_deliveries(d):
    groups = defaultdict(list)
    for m in new_messages(d):
        groups[m["external_id"]].append(m)
    dups = [g for g in groups.values() if len(g) > 1]
    assert 3 <= len(dups) <= 5
    for a, b in dups:
        assert a["seen_at"] < b["seen_at"]
        strip = lambda m: {k: v for k, v in m.items() if k not in ("label", "seen_at")}  # noqa: E731
        assert strip(a) == strip(b)


def test_new_times_before_now(d):
    for m in new_messages(d):
        assert repo.parse_time(m["sent_at"]) < NOW, m["label"]
        assert repo.parse_time(m["seen_at"]) < NOW, m["label"]
        assert repo.parse_time(m["sent_at"]) >= repo.parse_time("2026-11-02T00:00:00+05:30")


def test_only_c32_mentions_priyas_flight(d):
    hits = [m["label"] for m in d["messages"] if "6E 2134" in m["body"]]
    assert hits == ["c32"]


def test_calendar_events(d):
    events = [m for m in d["messages"] if m["app"] == "calendar"]
    assert len(events) == len({m["thread"] for m in events}) == 8
    by_start = {m["event"]["start"]: m for m in events}
    assert set(by_start) == {
        "2026-11-06T20:00:00+05:30",
        "2026-11-07T10:00:00+05:30",
        "2026-11-13T20:00:00+05:30",
        "2026-11-17T16:00:00+05:30",
        "2026-11-20T20:00:00+05:30",
        "2026-11-27T21:30:00+05:30",
        "2026-11-28T09:30:00+05:30",
        "2026-11-29T13:00:00+05:30",
    }
    stale = by_start["2026-11-20T20:00:00+05:30"]
    assert repo.parse_time(stale["event"]["start"]).strftime("%H:%M") == "20:00"
    assert repo.parse_time(stale["sent_at"]) < repo.parse_time("2026-11-18T10:15:00+05:30")
    assert "8:00 PM" in stale["body"]
    assert by_start["2026-11-28T09:30:00+05:30"]["event"]["attendees"] == ["You", "Sam", "Arjun"]
    for m in events:
        ev = m["event"]
        assert set(ev) == {"title", "start", "end", "location", "attendees", "status", "organizer"}
        assert ev["status"] == "confirmed"
        assert m["sender"] == ev["organizer"]
        assert m["body"].startswith(f"Calendar event: {ev['title']}.")
        assert repo.parse_time(m["sent_at"]) < repo.parse_time(ev["start"])
        assert "labels" not in m and "sender_email" not in m


def test_promos_are_bulk(d):
    msgs = promos(d)
    assert len({m["thread"] for m in msgs}) >= 20
    for m in msgs:
        assert m["body"].startswith("Subject: ")
        assert "Unsubscribe" in m["body"]
        assert m["sender"] != "You"
        assert not KNOWN_PEOPLE.search(m["body"]), m["label"]
        assert len(m["labels"]) == 1
    labels = Counter(m["labels"][0] for m in msgs)
    assert labels["CATEGORY_PROMOTIONS"] > labels["CATEGORY_UPDATES"] > 0
    assert labels["CATEGORY_SOCIAL"] > 0
    assert 0.05 <= labels["INBOX"] / len(msgs) <= 0.15


def test_distractors_present(d):
    subjects = Counter(m["body"].split("\n", 1)[0].removeprefix("Subject: ") for m in promos(d))
    kinds = Counter(x.kind for x in scenario_d.DISTRACTORS)
    assert set(kinds) == {"dinner_time", "weekend", "flight", "todo", "food"}
    assert all(2 <= n <= 4 for n in kinds.values())
    for x in scenario_d.DISTRACTORS:
        assert subjects[x.subject] >= 1, x.subject


def test_deterministic_build(tmp_path, d):
    outs = [tmp_path / "one.json", tmp_path / "two.json"]
    for out in outs:
        with pytest.raises(SystemExit) as exc:
            scenario_d.main(["build", "--out", str(out)])
        assert exc.value.code == 0
    assert outs[0].read_bytes() == outs[1].read_bytes()
    assert outs[0].read_bytes() == scenario_d.OUT_PATH.read_bytes()


def test_loads_with_store_schema(conn, d):
    repo.load_scenario(conn, copy.deepcopy(d))


def test_loads_through_repo(tmp_path, d):
    conn = store_conn(tmp_path / "d.db")
    try:
        result = repo.load_scenario(conn, copy.deepcopy(d))
        assert result.conversations_added == len(d["conversations"]) == 48
        assert result.messages_added == len(d["messages"])
        assert set(result.message_ids) == {m["label"] for m in d["messages"]}
    finally:
        conn.close()


def test_baseline_build_runs(tmp_path, d):
    from baseline.build import build

    db = tmp_path / "d.db"
    store_conn(db).close()
    written = build(db, d, FakeLLM(), FakeEmbedder(), log=lambda line: None)
    assert written == len(d["messages"])
