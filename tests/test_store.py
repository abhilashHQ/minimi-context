from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from helpers import T0, at, make_event

from person_context.models import Alias
from person_context.store import PersonExists, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.open(tmp_path / "db" / "context.db")


def test_migrations_and_wal(tmp_path: Path) -> None:
    path = tmp_path / "context.db"
    store = Store.open(path)
    assert store.schema_version() == 1
    store.close()
    reopened = Store.open(path)
    assert reopened.schema_version() == 1
    mode = sqlite3.connect(str(path)).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_create_person_has_confirmed_name_alias(store: Store) -> None:
    person = store.create_person("Asha Rao", T0)
    assert store.get_person(person.id) == person
    assert store.list_people() == [person]
    assert store.aliases(person.id) == [Alias(person.id, "Asha Rao", "name", "confirmed", None)]


def test_create_person_refuses_duplicate_name(store: Store) -> None:
    person = store.create_person("Asha Rao", T0)
    with pytest.raises(PersonExists) as info:
        store.create_person("  asha  rao ", T0)
    assert info.value.person_id == person.id


def test_save_compile_is_idempotent_and_sets_watermark(store: Store) -> None:
    person = store.create_person("Asha Rao", T0)
    events = [make_event(person.id, "first", occurred_at=at(1)), make_event(person.id, "second")]
    proposal = Alias(person.id, "asha@example.com", "email", "proposed", "asha@example.com")
    assert store.save_compile(person.id, events, [proposal], at(2)) == (2, 1)
    assert store.save_compile(person.id, events, [proposal], at(3)) == (0, 0)
    assert store.event_count(person.id) == 2
    assert store.watermark(person.id) == at(3)


def test_rejected_alias_is_not_proposed_again(store: Store) -> None:
    person = store.create_person("Asha Rao", T0)
    proposal = Alias(person.id, "Acme", "company", "proposed", "Asha from Acme")
    store.save_compile(person.id, [], [proposal], at(1))
    assert store.set_alias_status(person.id, "acme", "rejected")
    store.save_compile(person.id, [], [proposal], at(2))
    statuses = {a.value: a.status for a in store.aliases(person.id)}
    assert statuses == {"Asha Rao": "confirmed", "Acme": "rejected"}
    assert not store.set_alias_status(person.id, "Acme", "confirmed")


def test_timeline_newest_capped_and_oldest_first(store: Store) -> None:
    person = store.create_person("Asha Rao", T0)
    events = [make_event(person.id, f"quote {i}", occurred_at=at(i)) for i in range(5)]
    store.save_compile(person.id, list(reversed(events)), [], at(10))
    assert [e.source_quote for e in store.timeline(person.id, 3)] == [
        "quote 2",
        "quote 3",
        "quote 4",
    ]
    assert store.timeline(person.id, 10) == events


def test_seen_and_card_times(store: Store) -> None:
    person = store.create_person("Asha Rao", T0)
    assert store.seen(person.id) == (None, None)
    store.set_last_seen(person.id, at(1))
    store.set_last_card(person.id, at(2))
    store.set_last_seen(person.id, at(3))
    assert store.seen(person.id) == (at(3), at(2))


def test_mcp_call_counting(store: Store) -> None:
    store.record_mcp_call("search_memory", at(0))
    store.record_mcp_call("search_memory", at(5))
    assert store.mcp_calls_since(at(1)) == 1
    assert store.mcp_calls_since(at(0)) == 2


def test_failed_transaction_rolls_back(store: Store) -> None:
    person = store.create_person("Asha Rao", T0)
    bad = make_event(person.id, "x")
    bad = bad.__class__(**{**bad.__dict__, "speaker": "nobody"})
    with pytest.raises(sqlite3.IntegrityError):
        store.save_compile(person.id, [make_event(person.id, "ok"), bad], [], at(1))
    assert store.event_count(person.id) == 0
    assert store.watermark(person.id) is None


def test_each_thread_uses_its_own_connection(tmp_path: Path) -> None:
    path = tmp_path / "context.db"
    store = Store.open(path)
    person = store.create_person("Asha Rao", T0)
    results: list[int] = []

    def worker() -> None:
        own = Store.open(path)
        own.set_last_seen(person.id, at(1))
        results.append(own.event_count(person.id))
        own.close()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert results == [0]
    assert store.seen(person.id) == (at(1), None)
