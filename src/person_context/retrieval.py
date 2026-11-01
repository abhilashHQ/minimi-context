"""Retrievers choose which stored events a card or answer is built from."""

from __future__ import annotations

from typing import Protocol

from .models import Event
from .screen import ScreenContext
from .store import Store

TIMELINE_CAP = 300


class Retriever(Protocol):
    def retrieve(
        self, person_id: int, question: str, screen: ScreenContext | None = None
    ) -> list[Event]: ...


class WholeTimelineRetriever:
    """The person's newest TIMELINE_CAP events, oldest first, whatever the question."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def retrieve(
        self, person_id: int, question: str, screen: ScreenContext | None = None
    ) -> list[Event]:
        return self._store.timeline(person_id, TIMELINE_CAP)
