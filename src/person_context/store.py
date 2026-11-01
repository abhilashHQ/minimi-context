"""SQLite storage. All SQL lives in this module.

A Store wraps one connection and must only be used by the thread that opened it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from .models import Alias, AliasKind, AliasStatus, Event, Person, Speaker, normalize

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE person (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        display_name TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE alias (
        person_id INTEGER NOT NULL REFERENCES person(id),
        value TEXT NOT NULL,
        normalized TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('name', 'email', 'phone', 'company')),
        status TEXT NOT NULL CHECK (status IN ('proposed', 'confirmed', 'rejected')),
        evidence_quote TEXT,
        PRIMARY KEY (person_id, normalized)
    );
    CREATE INDEX alias_normalized ON alias(normalized);
    CREATE TABLE event (
        id TEXT PRIMARY KEY,
        person_id INTEGER NOT NULL REFERENCES person(id),
        occurred_at TEXT,
        source_app TEXT NOT NULL,
        source_thread TEXT NOT NULL,
        speaker TEXT NOT NULL CHECK (speaker IN ('self', 'them', 'other', 'unknown')),
        text TEXT NOT NULL,
        source_quote TEXT NOT NULL,
        ingested_at TEXT NOT NULL
    );
    CREATE INDEX event_person_time ON event(person_id, occurred_at);
    CREATE TABLE person_seen (
        person_id INTEGER PRIMARY KEY REFERENCES person(id),
        last_seen_at TEXT,
        last_card_at TEXT
    );
    CREATE TABLE compile_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE mcp_calls (
        called_at TEXT NOT NULL,
        tool TEXT NOT NULL
    );
    CREATE INDEX mcp_calls_time ON mcp_calls(called_at);
    """,
)


class PersonExists(Exception):
    def __init__(self, person_id: int) -> None:
        super().__init__(f"already tracked as person {person_id}")
        self.person_id = person_id


def _ts(value: datetime) -> str:
    # Fixed-width UTC strings so that text order equals time order.
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _watermark_key(person_id: int) -> str:
    return f"watermark:{person_id}"


class Store:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, path: Path) -> Store:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), isolation_level=None, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        store = cls(conn)
        store._migrate()
        return store

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _migrate(self) -> None:
        with self._tx() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            for number, script in enumerate(_MIGRATIONS[version:], start=version + 1):
                for statement in script.split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {number}")

    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    # People and aliases

    def create_person(self, display_name: str, now: datetime) -> Person:
        """Create a person with a confirmed name alias equal to the display name."""
        normalized = normalize(display_name)
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT person_id FROM alias WHERE normalized = ? AND status != 'rejected'",
                (normalized,),
            ).fetchone()
            if existing is not None:
                raise PersonExists(int(existing["person_id"]))
            cursor = conn.execute(
                "INSERT INTO person (display_name, created_at) VALUES (?, ?)",
                (display_name, _ts(now)),
            )
            person_id = cast(int, cursor.lastrowid)
            conn.execute(
                "INSERT INTO alias (person_id, value, normalized, kind, status, evidence_quote)"
                " VALUES (?, ?, ?, 'name', 'confirmed', NULL)",
                (person_id, display_name, normalized),
            )
        return Person(person_id, display_name, now)

    def get_person(self, person_id: int) -> Person | None:
        row = self._conn.execute("SELECT * FROM person WHERE id = ?", (person_id,)).fetchone()
        return None if row is None else _person(row)

    def list_people(self) -> list[Person]:
        rows = self._conn.execute("SELECT * FROM person ORDER BY id").fetchall()
        return [_person(row) for row in rows]

    def aliases(self, person_id: int) -> list[Alias]:
        rows = self._conn.execute(
            "SELECT * FROM alias WHERE person_id = ? ORDER BY rowid", (person_id,)
        ).fetchall()
        return [_alias(row) for row in rows]

    def all_aliases(self) -> list[Alias]:
        rows = self._conn.execute("SELECT * FROM alias ORDER BY person_id, rowid").fetchall()
        return [_alias(row) for row in rows]

    def aliases_matching(self, value: str) -> list[Alias]:
        """Aliases of any person whose normalized value equals normalize(value)."""
        rows = self._conn.execute(
            "SELECT * FROM alias WHERE normalized = ? ORDER BY person_id",
            (normalize(value),),
        ).fetchall()
        return [_alias(row) for row in rows]

    def set_alias_status(self, person_id: int, value: str, status: AliasStatus) -> bool:
        """Confirm or reject a proposed alias. Returns False if no proposed alias matched."""
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE alias SET status = ?"
                " WHERE person_id = ? AND normalized = ? AND status = 'proposed'",
                (status, person_id, normalize(value)),
            )
        return cursor.rowcount > 0

    # Compile output

    def save_compile(
        self,
        person_id: int,
        events: Sequence[Event],
        proposals: Sequence[Alias],
        watermark: datetime,
    ) -> tuple[int, int]:
        """Store events and alias proposals, then advance the watermark, in one transaction.

        Returns (events added, proposals added). Existing events and aliases, including
        rejected ones, are left untouched.
        """
        events_added = 0
        proposals_added = 0
        with self._tx() as conn:
            for event in events:
                cursor = conn.execute(
                    "INSERT INTO event (id, person_id, occurred_at, source_app,"
                    " source_thread, speaker, text, source_quote, ingested_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
                    (
                        event.id,
                        event.person_id,
                        None if event.occurred_at is None else _ts(event.occurred_at),
                        event.source_app,
                        event.source_thread,
                        event.speaker,
                        event.text,
                        event.source_quote,
                        _ts(event.ingested_at),
                    ),
                )
                events_added += cursor.rowcount
            for alias in proposals:
                cursor = conn.execute(
                    "INSERT INTO alias (person_id, value, normalized, kind, status, evidence_quote)"
                    " VALUES (?, ?, ?, ?, 'proposed', ?)"
                    " ON CONFLICT(person_id, normalized) DO NOTHING",
                    (
                        alias.person_id,
                        alias.value,
                        normalize(alias.value),
                        alias.kind,
                        alias.evidence_quote,
                    ),
                )
                proposals_added += cursor.rowcount
            conn.execute(
                "INSERT INTO compile_state (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_watermark_key(person_id), _ts(watermark)),
            )
        return events_added, proposals_added

    def watermark(self, person_id: int) -> datetime | None:
        row = self._conn.execute(
            "SELECT value FROM compile_state WHERE key = ?", (_watermark_key(person_id),)
        ).fetchone()
        return None if row is None else _dt(row["value"])

    # Events

    def timeline(self, person_id: int, limit: int) -> list[Event]:
        """The newest `limit` events of a person, returned oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM event WHERE person_id = ?"
            " ORDER BY COALESCE(occurred_at, ingested_at) DESC, id LIMIT ?",
            (person_id, limit),
        ).fetchall()
        return [_event(row) for row in reversed(rows)]

    def event_count(self, person_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM event WHERE person_id = ?", (person_id,)
        ).fetchone()
        return int(row[0])

    # Presence

    def seen(self, person_id: int) -> tuple[datetime | None, datetime | None]:
        """(last_seen_at, last_card_at) for a person."""
        row = self._conn.execute(
            "SELECT last_seen_at, last_card_at FROM person_seen WHERE person_id = ?",
            (person_id,),
        ).fetchone()
        if row is None:
            return None, None
        return _dt(row["last_seen_at"]), _dt(row["last_card_at"])

    def set_last_seen(self, person_id: int, when: datetime) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO person_seen (person_id, last_seen_at) VALUES (?, ?)"
                " ON CONFLICT(person_id) DO UPDATE SET last_seen_at = excluded.last_seen_at",
                (person_id, _ts(when)),
            )

    def set_last_card(self, person_id: int, when: datetime) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO person_seen (person_id, last_card_at) VALUES (?, ?)"
                " ON CONFLICT(person_id) DO UPDATE SET last_card_at = excluded.last_card_at",
                (person_id, _ts(when)),
            )

    # MCP call budget

    def record_mcp_call(self, tool: str, when: datetime) -> None:
        with self._tx() as conn:
            conn.execute("INSERT INTO mcp_calls (called_at, tool) VALUES (?, ?)", (_ts(when), tool))

    def mcp_calls_since(self, since: datetime) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM mcp_calls WHERE called_at >= ?", (_ts(since),)
        ).fetchone()
        return int(row[0])


def _person(row: sqlite3.Row) -> Person:
    return Person(
        id=int(row["id"]),
        display_name=str(row["display_name"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _alias(row: sqlite3.Row) -> Alias:
    return Alias(
        person_id=int(row["person_id"]),
        value=str(row["value"]),
        kind=cast(AliasKind, row["kind"]),
        status=cast(AliasStatus, row["status"]),
        evidence_quote=row["evidence_quote"],
    )


def _event(row: sqlite3.Row) -> Event:
    return Event(
        id=str(row["id"]),
        person_id=int(row["person_id"]),
        occurred_at=_dt(row["occurred_at"]),
        source_app=str(row["source_app"]),
        source_thread=str(row["source_thread"]),
        speaker=cast(Speaker, row["speaker"]),
        text=str(row["text"]),
        source_quote=str(row["source_quote"]),
        ingested_at=datetime.fromisoformat(row["ingested_at"]),
    )
