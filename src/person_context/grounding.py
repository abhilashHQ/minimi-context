"""Quote verification and the on-screen filter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .models import Event, normalize


def contains_quote(text: str, quote: str) -> bool:
    """True if the non-empty quote appears in text after normalize."""
    needle = normalize(quote)
    return bool(needle) and needle in normalize(text)


def quote_in_any(quote: str, texts: Iterable[str]) -> bool:
    return any(contains_quote(text, quote) for text in texts)


@dataclass(frozen=True)
class LineDraft:
    text: str
    event_ids: tuple[str, ...]
    quote: str


@dataclass(frozen=True)
class CitationDraft:
    event_id: str
    quote: str


@dataclass(frozen=True)
class GroundedLine:
    text: str
    quote: str
    event: Event


@dataclass(frozen=True)
class Citation:
    quote: str
    event: Event


def ground_lines(
    lines: Sequence[LineDraft], events: Mapping[str, Event], on_screen: Sequence[str]
) -> list[GroundedLine]:
    """Card lines whose event IDs are all known, whose quote is in one of those events'
    source_quote, and whose quote is not already in the on-screen messages."""
    screen_text = "\n".join(on_screen)
    grounded: list[GroundedLine] = []
    for line in lines:
        if not line.event_ids or any(ref not in events for ref in line.event_ids):
            continue
        source = next(
            (
                events[ref]
                for ref in line.event_ids
                if contains_quote(events[ref].source_quote, line.quote)
            ),
            None,
        )
        if source is None or contains_quote(screen_text, line.quote):
            continue
        grounded.append(GroundedLine(line.text, line.quote, source))
    return grounded


def ground_citations(
    citations: Sequence[CitationDraft], events: Mapping[str, Event]
) -> list[Citation]:
    """Citations whose event ID is known and whose quote is in that event's source_quote."""
    grounded: list[Citation] = []
    for citation in citations:
        event = events.get(citation.event_id)
        if event is not None and contains_quote(event.source_quote, citation.quote):
            grounded.append(Citation(citation.quote, event))
    return grounded
