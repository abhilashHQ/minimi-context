"""Pure card gate: decides whether returning to a chat should retrieve context."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import Config
from .models import Alias, Person, normalize


@dataclass(frozen=True)
class HeaderMatch:
    """A tracked person whose non-rejected alias equals the chat header."""

    header: str
    person: Person
    aliases: Sequence[Alias]


@dataclass(frozen=True)
class Decision:
    fire: bool
    reason: str


def should_retrieve(
    person: HeaderMatch | None,
    now: datetime,
    last_seen_at: datetime | None,
    last_card_at: datetime | None,
    cfg: Config,
) -> Decision:
    if person is None:
        return Decision(False, "untracked")
    header = normalize(person.header)
    if not any(a.status == "confirmed" and normalize(a.value) == header for a in person.aliases):
        return Decision(False, "unconfirmed")
    if last_seen_at is not None and now - last_seen_at < timedelta(hours=cfg.gap_hours):
        return Decision(False, "no_gap")
    if last_card_at is not None and now - last_card_at < timedelta(hours=cfg.cooldown_hours):
        return Decision(False, "cooldown")
    return Decision(True, "returned")
