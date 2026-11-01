from __future__ import annotations

from pathlib import Path

from helpers import T0, at, make_config

from person_context.models import Alias, Person
from person_context.trigger import HeaderMatch, should_retrieve

CFG = make_config(Path("unused.db"), gap_hours=6.0, cooldown_hours=2.0)
PERSON = Person(1, "Bhavya", T0)
CONFIRMED = HeaderMatch("Bhavya", PERSON, (Alias(1, "Bhavya", "name", "confirmed", None),))


def test_first_return_fires() -> None:
    decision = should_retrieve(CONFIRMED, at(1), None, None, CFG)
    assert decision.fire and decision.reason == "returned"


def test_gap() -> None:
    assert not should_retrieve(CONFIRMED, at(10), at(5), None, CFG).fire
    assert should_retrieve(CONFIRMED, at(11), at(5), None, CFG).fire
    assert should_retrieve(
        CONFIRMED, at(10.5), at(5), None, make_config(Path("x"), gap_hours=5.5)
    ).fire


def test_cooldown() -> None:
    decision = should_retrieve(CONFIRMED, at(20), at(10), at(19), CFG)
    assert not decision.fire and decision.reason == "cooldown"
    assert should_retrieve(CONFIRMED, at(21), at(10), at(19), CFG).fire


def test_untracked() -> None:
    decision = should_retrieve(None, at(1), None, None, CFG)
    assert not decision.fire and decision.reason == "untracked"


def test_unconfirmed_alias_does_not_fire() -> None:
    proposed = HeaderMatch(
        "Bhavs",
        PERSON,
        (
            Alias(1, "Bhavya", "name", "confirmed", None),
            Alias(1, "Bhavs", "name", "proposed", "Bhavs here"),
        ),
    )
    decision = should_retrieve(proposed, at(1), None, None, CFG)
    assert not decision.fire and decision.reason == "unconfirmed"


def test_repeat_return_inside_gap_does_nothing() -> None:
    # The caller updates last_seen_at on every tick while the chat is on screen.
    assert should_retrieve(CONFIRMED, at(10), None, None, CFG).fire
    assert not should_retrieve(CONFIRMED, at(10.001), at(10), at(10), CFG).fire
    assert not should_retrieve(CONFIRMED, at(12), at(10.5), at(10), CFG).fire
