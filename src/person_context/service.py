"""Orchestrates the card and Ask. No UI imports; safe to call from any worker thread."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from .compile import CompileLocked, CompileResult, OpenMinimi, compile_all, refresh_person
from .config import Config
from .grounding import (
    Citation,
    CitationDraft,
    GroundedLine,
    LineDraft,
    ground_citations,
    ground_lines,
)
from .identity import pending_aliases, resolve
from .llm import GeminiClient, LLMClient
from .minimi import CallBudget, StdioMinimiClient
from .models import Alias, Event, Person
from .privacy import gate_patterns, mentions
from .prompts import (
    ASK_SCHEMA,
    ASK_SYSTEM,
    CARD_SCHEMA,
    CARD_SYSTEM,
    ask_prompt,
    card_prompt,
    event_ref,
    screen_text,
)
from .retrieval import Retriever, WholeTimelineRetriever
from .screen import ScreenContext
from .store import Store
from .trigger import HeaderMatch, header_confirmed, should_retrieve

log = logging.getLogger(__name__)

CARD_MAX_LINES = 4
# minimi's connector is metered per week across all clients, so a card or Ask fetches
# fresh memories for a person at most this often.
REFRESH_MINUTES = 10
NO_CONTEXT = "I don't have that in this person's context."
UNKNOWN_PERSON = "I don't know which person you mean. Name someone you track."
ASK_FAILED = "Couldn't answer right now. Try again in a moment."


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Source:
    quote: str
    source_app: str
    occurred_at: datetime | None


@dataclass(frozen=True)
class CardLine:
    text: str
    source: Source


@dataclass(frozen=True)
class Card:
    person: Person
    lines: tuple[CardLine, ...]
    pending_alias: Alias | None


@dataclass(frozen=True)
class AskResult:
    status: str  # "answered", "no_context", "ambiguous", "unknown_person" or "failed"
    answer: str
    person: Person | None
    sources: tuple[Source, ...]


class Service:
    def __init__(
        self,
        cfg: Config,
        open_store: Callable[[], Store],
        llm: LLMClient,
        open_minimi: OpenMinimi,
        retriever: Callable[[Store], Retriever] = WholeTimelineRetriever,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._cfg = cfg
        self._open_store = open_store
        self._llm = llm
        self._open_minimi = open_minimi
        self._retriever = retriever
        self._clock = clock
        self._local = threading.local()

    def _store(self) -> Store:
        store: Store | None = getattr(self._local, "store", None)
        if store is None:
            store = self._open_store()
            self._local.store = store
        return store

    # Card

    def on_screen(self, screen: ScreenContext) -> Card | None:
        """Record that a chat is on screen and build a card if this is a return."""
        store = self._store()
        now = self._clock()
        match = match_header(store, screen.person_name)
        last_seen, last_card = store.seen(match.person.id) if match else (None, None)
        decision = should_retrieve(match, now, last_seen, last_card, self._cfg)
        if match is not None:
            store.set_last_seen(match.person.id, now)
        if not decision.fire or match is None:
            log.debug("trigger fire=false reason=%s", decision.reason)
            return None
        log.info("trigger person=%d fire=true reason=%s", match.person.id, decision.reason)
        try:
            return self._card(store, match.person, screen, now)
        except Exception as exc:
            log.error("card failed person=%d error=%s", match.person.id, type(exc).__name__)
            return None

    def _card(
        self, store: Store, person: Person, screen: ScreenContext, now: datetime
    ) -> Card | None:
        self._refresh(store, person, now)
        events = self._retriever(store).retrieve(person.id, screen_text(screen), screen)
        if not events:
            log.info("card person=%d shown=false reason=no_events", person.id)
            return None
        raw = self._llm.generate_json(
            self._cfg.model_answer,
            CARD_SYSTEM,
            card_prompt(person, events, screen, now),
            CARD_SCHEMA,
        )
        drafts = [
            LineDraft(line["text"], tuple(line["event_ids"]), line["quote"])
            for line in raw["lines"]
        ]
        grounded = ground_lines(drafts, _refs(events), [m.text for m in screen.messages])
        lines = tuple(_card_line(line) for line in grounded[:CARD_MAX_LINES])
        log.info(
            "card person=%d show=%s lines_proposed=%d lines_kept=%d",
            person.id,
            raw["show"],
            len(drafts),
            len(lines),
        )
        if not raw["show"] or not lines:
            return None
        store.set_last_card(person.id, now)
        pending = pending_aliases(store, person.id)
        return Card(person, lines, pending[0] if pending else None)

    # Ask

    def ask(self, question: str, screen: ScreenContext | None) -> AskResult:
        store = self._store()
        now = self._clock()
        target = _ask_target(store, question, screen)
        if isinstance(target, AskResult):
            log.info("ask status=%s", target.status)
            return target
        try:
            result = self._answer(store, target, question, screen, now)
        except Exception as exc:
            log.error("ask failed person=%d error=%s", target.id, type(exc).__name__)
            return AskResult("failed", ASK_FAILED, target, ())
        log.info(
            "ask person=%d status=%s sources=%d", target.id, result.status, len(result.sources)
        )
        return result

    def _answer(
        self,
        store: Store,
        person: Person,
        question: str,
        screen: ScreenContext | None,
        now: datetime,
    ) -> AskResult:
        self._refresh(store, person, now)
        events = self._retriever(store).retrieve(person.id, question, screen)
        if not events:
            return AskResult("no_context", NO_CONTEXT, person, ())
        raw = self._llm.generate_json(
            self._cfg.model_answer,
            ASK_SYSTEM,
            ask_prompt(person, events, question, now),
            ASK_SCHEMA,
        )
        drafts = [CitationDraft(c["event_id"], c["quote"]) for c in raw["citations"]]
        citations = ground_citations(drafts, _refs(events))
        if not citations:
            return AskResult("no_context", NO_CONTEXT, person, ())
        return AskResult("answered", raw["answer"], person, tuple(_source(c) for c in citations))

    # Compile and identity

    def compile(self) -> list[CompileResult] | None:
        """Compile everyone. None if another compile is running."""
        try:
            return compile_all(
                self._store(), self._open_minimi, self._llm, self._cfg, self._clock()
            )
        except CompileLocked:
            log.info("compile skipped reason=locked")
            return None

    def resolve_alias(self, person_id: int, value: str, *, confirm: bool) -> bool:
        return resolve(self._store(), person_id, value, confirm=confirm)

    def _refresh(self, store: Store, person: Person, now: datetime) -> None:
        """Freshness: compile this person since the last compile; failures keep stored events.

        Throttled to one attempt per REFRESH_MINUTES. The attempt is recorded whatever its
        outcome, so a quota failure is not repeated on every Ask.
        """
        last = store.last_refresh(person.id)
        if last is not None and now - last < timedelta(minutes=REFRESH_MINUTES):
            log.info("refresh skipped person=%d reason=throttled", person.id)
            return
        store.set_last_refresh(person.id, now)
        try:
            refresh_person(store, self._open_minimi, self._llm, self._cfg, person, now)
        except CompileLocked:
            log.info("refresh skipped person=%d reason=locked", person.id)
        except Exception as exc:
            log.warning("refresh failed person=%d error=%s", person.id, type(exc).__name__)


def describe_source(source: Source) -> str:
    """The quote with its app and date, as shown on the card and in answers."""
    parts = [f"\u201c{source.quote}\u201d"]
    if source.source_app:
        parts.append(source.source_app)
    if source.occurred_at is not None:
        parts.append(source.occurred_at.strftime("%d %b %Y"))
    return " \u00b7 ".join(parts)


def build_service(cfg: Config) -> Service:
    def open_minimi(store: Store) -> StdioMinimiClient:
        budget = CallBudget(store, cfg.mcp_weekly_budget, utcnow)
        return StdioMinimiClient(cfg.minimi_command, cfg.minimi_env, budget)

    return Service(
        cfg,
        open_store=lambda: Store.open(cfg.db_path),
        llm=GeminiClient(cfg.gemini_api_key),
        open_minimi=open_minimi,
    )


def match_header(store: Store, header: str) -> HeaderMatch | None:
    """The one tracked person with a non-rejected alias equal to the header.

    If several people match, only a single confirmed match counts.
    """
    matches = [a for a in store.aliases_matching(header) if a.status != "rejected"]
    people = {a.person_id for a in matches}
    if len(people) > 1:
        people = {a.person_id for a in matches if a.status == "confirmed"}
        if len(people) != 1:
            log.info("header ambiguous candidates=%d", len({a.person_id for a in matches}))
            return None
    if not people:
        return None
    person = store.get_person(people.pop())
    if person is None:
        return None
    return HeaderMatch(header, person, tuple(store.aliases(person.id)))


def _ask_target(store: Store, question: str, screen: ScreenContext | None) -> Person | AskResult:
    named: dict[int, Person] = {}
    for alias in store.all_aliases():
        if alias.person_id not in named and mentions(question, gate_patterns([alias])):
            person = store.get_person(alias.person_id)
            if person is not None:
                named[person.id] = person
    if len(named) > 1:
        names = ", ".join(sorted(p.display_name for p in named.values()))
        return AskResult("ambiguous", f"That matches more than one person: {names}.", None, ())
    if named:
        return next(iter(named.values()))
    match = match_header(store, screen.person_name) if screen is not None else None
    if match is not None and header_confirmed(match):
        return match.person
    return AskResult("unknown_person", UNKNOWN_PERSON, None, ())


def _refs(events: Sequence[Event]) -> dict[str, Event]:
    return {event_ref(event): event for event in events}


def _source(citation: Citation) -> Source:
    return Source(citation.quote, citation.event.source_app, citation.event.occurred_at)


def _card_line(line: GroundedLine) -> CardLine:
    return CardLine(line.text, Source(line.quote, line.event.source_app, line.event.occurred_at))
