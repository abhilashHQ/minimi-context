from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from harness.ingest import Bursts, gate_reason, ingest, to_source
from harness.retrieve import time_window
from harness.store import Store
from harness.text import (
    human_time, overlap, parse_event_at, quote_ok, rewrite_dates, snap_weekday, weekday_in,
)

IST = "+05:30"


def msg(label, sent, sender="Sam", thread="sam", app="whatsapp_web", body="hi", **kw):
    return {"label": label, "app": app, "thread": thread, "sender": sender,
            "sent_at": f"2026-11-{sent}:00{IST}", "body": body, **kw}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "h.db")


def test_people_match_by_first_name_prefix():
    assert overlap(["Sam"], ["Samuel Rao"])
    assert overlap(["Mr. Kapoor"], ["Kapoor"])
    assert not overlap(["Sam"], ["Priya Nair"])
    assert not overlap(["You"], ["you"])


def test_weekday_snap_moves_to_next_named_day():
    # "brunch Saturday" said Tue Nov 24, model wrote Thu Nov 26
    out = snap_weekday(f"2026-11-26T09:30:00{IST}", "And brunch Saturday at 9:30 with Arjun?", f"2026-11-24T18:25:00{IST}")
    assert out == f"2026-11-28T09:30:00{IST}"
    assert snap_weekday(f"2026-11-28T09:30:00{IST}", "brunch Saturday", f"2026-11-24T18:25:00{IST}") is None
    assert snap_weekday(f"2026-11-28T09:30:00{IST}", "Yes, see you then", f"2026-11-24T18:25:00{IST}") is None
    assert weekday_in("Monday to Sunday") is None


def test_rewrite_dates_in_memory_text():
    assert rewrite_dates("Brunch on Thu Nov 26 at 9:30 AM", f"2026-11-26T09:30:00{IST}",
                         f"2026-11-28T09:30:00{IST}") == "Brunch on Sat Nov 28 at 9:30 AM"


def test_event_at_parsing_and_due_dates():
    assert parse_event_at("2026-11-27T21:30", f"2026-11-24T18:22:00{IST}") == f"2026-11-27T21:30:00{IST}"
    due = parse_event_at("2026-11-30", f"2026-11-20T10:05:00{IST}")
    assert due == f"2026-11-30T23:59:00{IST}"
    assert human_time(due) == "Mon Nov 30"
    assert human_time(f"2026-11-27T21:30:00{IST}") == "Fri Nov 27, 9:30 PM"
    assert parse_event_at("soon", f"2026-11-20T10:05:00{IST}") is None


def test_quote_must_come_from_message():
    assert quote_ok("Here is the deck.", "Here is the deck.")
    assert quote_ok("here is the deck", "Here is the deck.")
    assert not quote_ok("I sent the cap table", "Here is the deck.")
    assert not quote_ok("", "Here is the deck.")


def test_mail_copy_from_connector_is_a_duplicate(store):
    b = Bursts(store)
    titles = {"friday-dinner": "Friday dinner"}
    screen = msg("c01", "18T10:15", sender="Samuel Rao", thread="friday-dinner", app="gmail", body="Can we do 9:30?")
    copy = {**screen, "label": "d1", "app": "gmail_connector", "body": "Can we do 9:30?\n\nSam",
            "seen_at": f"2026-11-18T10:20:00{IST}", "labels": ["INBOX"]}
    assert ingest(store, b, to_source(screen, titles))[1] == "added"
    assert ingest(store, b, to_source(copy, titles))[1] == "duplicate"


def test_promo_mail_is_gated(store):
    titles = {"promo-olive": "Olive"}
    promo = msg("d2", "20T09:00", sender="Olive", thread="promo-olive", app="gmail_connector",
                body="Friday dinner for two at 8 PM", labels=["CATEGORY_PROMOTIONS"], sender_email="offers@olive.in")
    src = to_source(promo, titles)
    assert gate_reason(store, src) == "bulk mail, never replied"
    assert ingest(store, Bursts(store), src)[1] == "gated"


def test_bursts_close_on_quiet_span_and_count(store):
    b = Bursts(store)
    titles = {"sam": "Sam"}
    feed = [msg("a", "02T19:05"), msg("b", "02T19:07", sender="You"), msg("c", "02T20:30")]
    ids = []
    for m in feed:
        src = to_source(m, titles)
        b.tick(src["seen_at"])
        sid, _ = ingest(store, b, src)
        ids.append(store.source(sid)["burst_id"])
    assert ids[0] == ids[1] != ids[2]
    assert b.ready() == [ids[0]]
    b.tick(f"2026-11-02T21:31:00{IST}")
    assert b.ready() == [ids[0], ids[2]]


def test_vectors_survive_reopen(tmp_path):
    s = Store(tmp_path / "v.db")
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    mid = s.add_memory({"text": "x", "kind": "fact", "valid_from": f"2026-11-02T19:05:00{IST}"}, v)
    s.commit()
    again = Store(tmp_path / "v.db")
    assert again.vector_top(v, None, 1)[0][0] == mid


def test_time_windows_read_against_now():
    now = datetime.fromisoformat(f"2026-11-26T20:00:00{IST}")  # Thursday
    (start, end), _ = time_window("what's on this weekend?", now)
    assert (start.date().isoformat(), end.date().isoformat()) == ("2026-11-28", "2026-11-29")
    (start, end), _ = time_window("what did i agree to last week, monday to sunday?", now)
    assert (start.date().isoformat(), end.date().isoformat()) == ("2026-11-16", "2026-11-22")
    (start, _), _ = time_window("dinner tomorrow", now)
    assert start.date().isoformat() == "2026-11-27"
    assert time_window("does sam eat fish", now) is None
