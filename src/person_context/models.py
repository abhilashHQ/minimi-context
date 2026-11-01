"""Core dataclasses and the single text normalizer."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Speaker = Literal["self", "them", "other", "unknown"]
AliasKind = Literal["name", "email", "phone", "company"]
AliasStatus = Literal["proposed", "confirmed", "rejected"]

SPEAKERS: tuple[Speaker, ...] = ("self", "them", "other", "unknown")
ALIAS_KINDS: tuple[AliasKind, ...] = ("name", "email", "phone", "company")

_DIRECTION_MARKS = {0x200E: None, 0x200F: None}
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC, strip U+200E and U+200F, collapse whitespace, casefold."""
    text = unicodedata.normalize("NFKC", text).translate(_DIRECTION_MARKS)
    return _WHITESPACE.sub(" ", text).strip().casefold()


def event_id(person_id: int, source_app: str, source_quote: str) -> str:
    key = f"{person_id}|{source_app}|{normalize(source_quote)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Person:
    id: int
    display_name: str
    created_at: datetime


@dataclass(frozen=True)
class Alias:
    person_id: int
    value: str
    kind: AliasKind
    status: AliasStatus
    evidence_quote: str | None


@dataclass(frozen=True)
class Event:
    id: str
    person_id: int
    occurred_at: datetime | None
    source_app: str
    source_thread: str
    speaker: Speaker
    text: str
    source_quote: str
    ingested_at: datetime
