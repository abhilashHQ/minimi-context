from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from helpers import T0, at, make_event

from person_context import store as store_module
from person_context.models import Alias
from person_context.store import PersonExists, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.open(tmp_path / "db" / "context.db")


def test_migrations_and_wal(tmp_path: Path) -> None:
    path = tmp_path / "context.db"
    store = Store.open(path)
    assert store.schema_version() == 3
    store.close()
    reopened = Store.open(path)
    assert reopened.schema_version() == 3
    mode = sqlite3.connect(str(path)).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_migrations_upgrade_an_existing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "context.db"
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "_MIGRATIONS", store_module._MIGRATIONS[:1])
        old = Store.open(path)
        assert old.schema_version() == 1
        person = old.create_person("Asha Rao", T0)
        old.close()
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "_MIGRATIONS", store_module._MIGRATIONS[:2])
        v2 = Store.open(path)
        assert v2.schema_version() == 2
        v2._conn.execute("INSERT INTO seen_block (person_id, hash) VALUES (?, 'old')", (person.id,))
        v2.close()
    upgraded = Store.open(path)
    assert upgraded.schema_version() == 3
    upgraded.save_compile(
        person.id, [], [], at(1), [("abc", "Asha A"), ("abc", "x"), ("def", "Asha D")]
    )
    assert upgraded.seen_block_hashes(person.id) == {"old", "abc", "def"}
    assert upgraded.recent_blocks(person.id, 20) == ["Asha A", "Asha D"]
    assert upgraded.list_people() == [person]


def test_only_the_newest_blocks_keep_text(store: Store) -> None:
    person = store.create_person("Asha Rao", T0)
    store.save_compile(person.id, [], [], at(1), [(f"h{i}", f"block {i}") for i in range(15)])
    store.save_compile(person.id, [], [], at(2), [(f"h{i}", f"block {i}") for i in range(15, 30)])
    newest = [f"block {i}" for i in range(10, 30)]
    assert store_module.BLOCK_CAP == 20
    assert store.recent_blocks(person.id, 20) == newest
    assert store.recent_blocks(person.id, 3) == newest[-3:]
    assert len(store.seen_block_hashes(person.id)) == 30
    nulls = store._conn.execute(
        "SELECT COUNT(*) FROM seen_block WHERE person_id = ? AND text IS NULL", (person.id,)
    ).fetchone()[0]
    assert nulls == 10


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
