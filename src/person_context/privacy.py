"""Alias gate. Runs before any minimi text can reach the LLM."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import Alias, normalize

GATE_STATUSES = frozenset({"confirmed", "proposed"})


@dataclass(frozen=True)
class GateResult:
    passed: tuple[str, ...]
    dropped: int


def _pattern(value: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(normalize(value))}(?!\w)")


def gate_patterns(aliases: Iterable[Alias]) -> list[re.Pattern[str]]:
    """Patterns for the aliases that may pass the gate: confirmed or proposed, non-empty."""
    return [
        _pattern(alias.value)
        for alias in aliases
        if alias.status in GATE_STATUSES and normalize(alias.value)
    ]


def mentions(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    normalized = normalize(text)
    return any(pattern.search(normalized) for pattern in patterns)


def gate(blocks: Sequence[str], aliases: Iterable[Alias]) -> GateResult:
    """Keep only blocks that contain one of the person's aliases on a word boundary."""
    patterns = gate_patterns(aliases)
    passed = tuple(block for block in blocks if mentions(block, patterns))
    return GateResult(passed=passed, dropped=len(blocks) - len(passed))
