"""Shared test builders. All names and data are made up."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from person_context.models import Event, Speaker, event_id

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def make_event(
    person_id: int,
    quote: str,
    *,
    text: str = "",
    occurred_at: datetime | None = None,
    source_app: str = "Mail",
    speaker: Speaker = "them",
) -> Event:
    return Event(
        id=event_id(person_id, source_app, quote),
        person_id=person_id,
        occurred_at=occurred_at,
        source_app=source_app,
        source_thread="thread",
        speaker=speaker,
        text=text or quote,
        source_quote=quote,
        ingested_at=T0,
    )
