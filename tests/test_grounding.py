from __future__ import annotations

from helpers import make_event

from person_context.grounding import (
    CitationDraft,
    LineDraft,
    contains_quote,
    ground_citations,
    ground_lines,
    quote_in_any,
)

LRM = "\N{LEFT-TO-RIGHT MARK}"


def test_contains_quote_normalizes_both_sides() -> None:
    assert contains_quote(f"{LRM}The order is 40   UNITS.", f"order is 40 units{LRM}")
    assert not contains_quote("The order is 40 units.", "order is 45 units")
    assert not contains_quote("anything", "  ")


def test_quote_in_any_block() -> None:
    assert quote_in_any("ships Friday", ["nothing here", "It ships friday morning"])
    assert not quote_in_any("ships Friday", [])


EVENT = make_event(1, "Asha: the order is now 55 units, delivery Friday")
OTHER = make_event(1, "Asha prefers calls after 4pm")
EVENTS = {"e1": EVENT, "e2": OTHER}


def test_ground_lines_keeps_verified_line_and_its_event() -> None:
    lines = [LineDraft("Order changed to 55", ("e2", "e1"), "order is now 55 units")]
    [line] = ground_lines(lines, EVENTS, ["hi", "any update?"])
    assert line.event == EVENT
    assert line.quote == "order is now 55 units"


def test_fabricated_quote_is_dropped() -> None:
    lines = [LineDraft("Order is 70", ("e1",), "order is now 70 units")]
    assert ground_lines(lines, EVENTS, []) == []


def test_unknown_or_missing_event_ids_are_dropped() -> None:
    lines = [
        LineDraft("x", (), "order is now 55 units"),
        LineDraft("x", ("e1", "e9"), "order is now 55 units"),
    ]
    assert ground_lines(lines, EVENTS, []) == []


def test_quote_already_on_screen_is_dropped() -> None:
    lines = [LineDraft("Calls after 4", ("e2",), "calls after 4pm")]
    screen = [f"{LRM}Your message, I know you prefer CALLS after 4pm, 10:05 AM, Read"]
    assert ground_lines(lines, EVENTS, screen) == []


def test_ground_citations() -> None:
    citations = [
        CitationDraft("e1", "delivery Friday"),
        CitationDraft("e2", "prefers emails"),
        CitationDraft("e7", "delivery Friday"),
    ]
    [citation] = ground_citations(citations, EVENTS)
    assert citation.event == EVENT and citation.quote == "delivery Friday"
