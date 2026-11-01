from __future__ import annotations

from person_context.models import event_id, normalize

LRM = "\N{LEFT-TO-RIGHT MARK}"
RLM = "\N{RIGHT-TO-LEFT MARK}"


def test_normalize_strips_direction_marks() -> None:
    assert normalize(f"{LRM}Your message, Hi{RLM}") == "your message, hi"


def test_normalize_collapses_whitespace_and_casefolds() -> None:
    assert normalize("  Stra\N{LATIN SMALL LETTER SHARP S}e \n\t  OK  ") == "strasse ok"


def test_normalize_nfkc() -> None:
    assert normalize("\N{LATIN SMALL LIGATURE FI}ne \N{FULLWIDTH DIGIT FOUR}0") == "fine 40"


def test_event_id_is_stable_across_quote_formatting() -> None:
    a = event_id(1, "Mail", "Order is 40 units")
    assert a == event_id(1, "Mail", f"{LRM}order  is 40 UNITS ")
    assert len(a) == 64
    assert a != event_id(2, "Mail", "Order is 40 units")
    assert a != event_id(1, "WhatsApp", "Order is 40 units")
