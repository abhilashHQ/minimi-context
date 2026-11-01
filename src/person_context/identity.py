"""Tracking people, alias proposals and their confirmation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from .grounding import contains_quote, quote_in_any
from .models import Alias, AliasKind, Person, normalize
from .store import Store


@dataclass(frozen=True)
class ProposalDraft:
    value: str
    kind: AliasKind
    evidence_quote: str


def track(store: Store, name: str, now: datetime) -> Person:
    """Create a person with a confirmed name alias. Raises PersonExists or ValueError."""
    display_name = " ".join(name.split())
    if not normalize(display_name):
        raise ValueError("name must not be empty")
    return store.create_person(display_name, now)


def pending_aliases(store: Store, person_id: int) -> list[Alias]:
    return [alias for alias in store.aliases(person_id) if alias.status == "proposed"]


def resolve(store: Store, person_id: int, value: str, *, confirm: bool) -> bool:
    """Confirm or reject a proposed alias. False if there was no such proposal."""
    return store.set_alias_status(person_id, value, "confirmed" if confirm else "rejected")


def accept_proposals(
    person_id: int,
    drafts: Sequence[ProposalDraft],
    blocks: Sequence[str],
    known: Sequence[Alias],
) -> list[Alias]:
    """Proposals whose evidence is grounded in the source blocks and names the value.

    Known values, including rejected ones, are never proposed again.
    """
    seen = {normalize(alias.value) for alias in known}
    accepted: list[Alias] = []
    for draft in drafts:
        value = " ".join(draft.value.split())
        key = normalize(value)
        if not key or key in seen:
            continue
        if not contains_quote(draft.evidence_quote, value):
            continue
        if not quote_in_any(draft.evidence_quote, blocks):
            continue
        seen.add(key)
        accepted.append(Alias(person_id, value, draft.kind, "proposed", draft.evidence_quote))
    return accepted
