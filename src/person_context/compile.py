"""Incremental compile: minimi memories through the privacy gate into stored events."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast

from .config import Config
from .grounding import quote_in_any
from .identity import ProposalDraft, accept_proposals
from .llm import LLMClient
from .minimi import MinimiClient, look_back, split_results
from .models import Alias, AliasKind, Event, Person, Speaker, event_id, normalize
from .privacy import gate
from .prompts import COMPILE_SCHEMA, COMPILE_SYSTEM, compile_prompt
from .store import Store

log = logging.getLogger(__name__)

FIRST_RUN_LOOK_BACK = timedelta(days=14)
# time_range filters on capture time, and minimi writes a memory a few minutes after
# its capture, so each look-back reaches an hour behind the watermark.
LOOK_BACK_OVERLAP_HOURS = 1
MAX_PAGES = 3

OpenMinimi = Callable[[Store], AbstractContextManager[MinimiClient]]


class CompileLocked(Exception):
    """Another compile is running."""


@dataclass(frozen=True)
class CompileResult:
    person_id: int
    outcome: str  # "stored", "nothing_new" or "failed"
    events_added: int = 0
    proposals_added: int = 0
    blocks_passed: int = 0
    blocks_dropped: int = 0
    events_dropped: int = 0
    error: str | None = None


@contextmanager
def compile_lock(db_path: Path) -> Iterator[None]:
    """Non-blocking exclusive lock shared by every process and thread using db_path."""
    path = db_path.with_name(db_path.name + ".compile.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CompileLocked() from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def compile_all(
    store: Store, open_minimi: OpenMinimi, llm: LLMClient, cfg: Config, run_started: datetime
) -> list[CompileResult]:
    """Compile every tracked person in one minimi batch. Raises CompileLocked."""
    with compile_lock(cfg.db_path), open_minimi(store) as minimi:
        return [
            compile_person(store, minimi, llm, cfg, person, run_started)
            for person in store.list_people()
        ]


def refresh_person(
    store: Store,
    open_minimi: OpenMinimi,
    llm: LLMClient,
    cfg: Config,
    person: Person,
    run_started: datetime,
) -> CompileResult:
    """Compile one person since the last compile. Raises CompileLocked."""
    with compile_lock(cfg.db_path), open_minimi(store) as minimi:
        return compile_person(store, minimi, llm, cfg, person, run_started)


def compile_person(
    store: Store,
    minimi: MinimiClient,
    llm: LLMClient,
    cfg: Config,
    person: Person,
    run_started: datetime,
) -> CompileResult:
    """Never raises. Any failure leaves the watermark unchanged so the next run retries."""
    started = time.monotonic()
    try:
        result = _compile(store, minimi, llm, cfg, person, run_started)
    except Exception as exc:
        log.warning("compile failed person=%d error=%s", person.id, type(exc).__name__)
        return CompileResult(person.id, "failed", error=type(exc).__name__)
    log.info(
        "compile person=%d outcome=%s events_added=%d proposals_added=%d blocks_passed=%d"
        " blocks_dropped=%d events_dropped=%d duration_ms=%d",
        person.id,
        result.outcome,
        result.events_added,
        result.proposals_added,
        result.blocks_passed,
        result.blocks_dropped,
        result.events_dropped,
        (time.monotonic() - started) * 1000,
    )
    return result


def _compile(
    store: Store,
    minimi: MinimiClient,
    llm: LLMClient,
    cfg: Config,
    person: Person,
    run_started: datetime,
) -> CompileResult:
    watermark = store.watermark(person.id)
    if watermark is None:
        hours = math.ceil(FIRST_RUN_LOOK_BACK.total_seconds() / 3600)
    else:
        elapsed = (run_started - watermark).total_seconds() / 3600
        hours = max(0, math.ceil(elapsed)) + LOOK_BACK_OVERLAP_HOURS
    aliases = [alias for alias in store.aliases(person.id) if alias.status == "confirmed"]

    blocks = _search(minimi, aliases, hours)
    gated = gate(blocks, aliases)
    seen = store.seen_block_hashes(person.id)
    new_blocks = [block for block in gated.passed if block_hash(block) not in seen]
    if not new_blocks:
        return CompileResult(person.id, "nothing_new", blocks_dropped=gated.dropped)

    raw = llm.generate_json(
        cfg.model_compile,
        COMPILE_SYSTEM,
        compile_prompt(person, aliases, new_blocks),
        COMPILE_SCHEMA,
    )
    events = [_event(person.id, item, run_started) for item in raw["events"]]
    grounded = [e for e in events if e.text.strip() and quote_in_any(e.source_quote, new_blocks)]
    proposals = accept_proposals(
        person.id,
        [
            ProposalDraft(item["value"], cast(AliasKind, item["kind"]), item["evidence_quote"])
            for item in raw["alias_proposals"]
        ],
        new_blocks,
        store.aliases(person.id),
    )
    events_added, proposals_added = store.save_compile(
        person.id, grounded, proposals, run_started, [(block_hash(b), b) for b in new_blocks]
    )
    return CompileResult(
        person.id,
        "stored",
        events_added=events_added,
        proposals_added=proposals_added,
        blocks_passed=len(new_blocks),
        blocks_dropped=gated.dropped,
        events_dropped=len(events) - len(grounded),
    )


def block_hash(block: str) -> str:
    return hashlib.sha256(normalize(block).encode("utf-8")).hexdigest()


def _search(minimi: MinimiClient, aliases: Sequence[Alias], hours: int) -> list[str]:
    """Blocks from search_memory for each distinct alias, up to MAX_PAGES pages each."""
    queries = list({normalize(alias.value): alias.value for alias in aliases}.values())
    blocks: dict[str, str] = {}
    for query in queries:
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            result = minimi.search_memory(query, look_back(hours), cursor=cursor)
            for block in split_results(result.text):
                blocks.setdefault(normalize(block), block)
            cursor = result.next_cursor
            if cursor is None:
                break
    return list(blocks.values())


def _event(person_id: int, item: Mapping[str, Any], ingested_at: datetime) -> Event:
    source_app = item["source_app"].strip()
    return Event(
        id=event_id(person_id, source_app, item["source_quote"]),
        person_id=person_id,
        occurred_at=parse_timestamp(item["occurred_at"]),
        source_app=source_app,
        source_thread=item["source_thread"].strip(),
        speaker=cast(Speaker, item["speaker"]),
        text=item["text"].strip(),
        source_quote=item["source_quote"],
        ingested_at=ingested_at,
    )


def parse_timestamp(value: str) -> datetime | None:
    """ISO 8601 to an aware datetime; naive times are taken as UTC. None if unparseable."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
