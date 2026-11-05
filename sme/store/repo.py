"""SQLite store: schema setup and scenario loading.

All times are Asia/Kolkata. India has no daylight saving time, so a fixed
+05:30 offset is exact and avoids needing the tz database on Windows.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30), "IST")
SELF_LABEL = "You"

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the store tables once. The schema file has no IF NOT EXISTS."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages'"
    ).fetchone()
    if not exists:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()


def normalize_body(body: str) -> str:
    return " ".join(body.split()).lower()


def parse_time(value: str) -> datetime:
    """Parse an ISO time that carries an offset and convert it to IST."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"time has no UTC offset: {value!r}")
    return dt.astimezone(IST)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(IST).isoformat(timespec="seconds")


def conversation_id(app: str, external_key: str) -> str:
    return hashlib.sha256(f"{app}|{external_key}".encode()).hexdigest()


def message_id(
    conv_id: str, sender_label: str, body_norm: str, sent_minute: str | None, occurrence: int
) -> str:
    key = f"{conv_id}|{sender_label}|{body_norm}|{sent_minute or ''}|{occurrence}"
    return hashlib.sha256(key.encode()).hexdigest()


def scenario_path(name: str) -> Path:
    return CORPUS_DIR / f"scenario_{name.lower()}.json"


def read_scenario(name_or_path: str | Path) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = scenario_path(str(name_or_path))
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class LoadResult:
    conversations_added: int = 0
    messages_added: int = 0
    # Corpus label (m1, m2, ...) -> message id, in ingest order.
    message_ids: dict[str, str] = field(default_factory=dict)


def load_scenario(
    conn: sqlite3.Connection, scenario: dict[str, Any], observed_at: str | None = None
) -> LoadResult:
    """Insert a scenario's conversations and messages in the order the file lists them.

    Loading the same scenario again adds nothing: ids are content hashes and
    existing rows only get their last_observed_at refreshed.
    """
    init_schema(conn)
    observed = to_iso(parse_time(observed_at or scenario["now"]))
    result = LoadResult()

    conv_ids: dict[str, str] = {}
    for conv in scenario["conversations"]:
        cid = conversation_id(conv["app"], conv["key"])
        conv_ids[f"{conv['app']}|{conv['key']}"] = cid
        cur = conn.execute(
            "INSERT OR IGNORE INTO conversations (id, app, external_key, title)"
            " VALUES (?, ?, ?, ?)",
            (cid, conv["app"], conv["key"], conv["title"]),
        )
        result.conversations_added += cur.rowcount

    seen: Counter[tuple[str, str, str, str | None]] = Counter()
    for msg in scenario["messages"]:
        cid = conv_ids[f"{msg['app']}|{msg['thread']}"]
        sender = msg["sender"]
        body = msg["body"]
        body_norm = normalize_body(body)
        date_source = msg["date_source"]

        if msg.get("sent_at"):
            if date_source == "unknown":
                raise ValueError(f"{msg.get('label')}: sent_at given but date_source is unknown")
            sent = parse_time(msg["sent_at"])
            sent_at = to_iso(sent)
            sent_minute = sent.strftime("%Y-%m-%dT%H:%M")
        else:
            if date_source != "unknown":
                raise ValueError(f"{msg.get('label')}: no sent_at but date_source is {date_source}")
            sent_at = sent_minute = None

        seen[(cid, sender, body_norm, sent_minute)] += 1
        occurrence = seen[(cid, sender, body_norm, sent_minute)]
        mid = message_id(cid, sender, body_norm, sent_minute, occurrence)
        direction = "out" if sender == SELF_LABEL else "in"

        cur = conn.execute(
            """INSERT OR IGNORE INTO messages
               (id, conversation_id, sender_label, direction, body, body_norm, sent_at,
                sent_minute, date_source, occurrence, is_partial, first_observed_at,
                last_observed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
            (
                mid,
                cid,
                sender,
                direction,
                body,
                body_norm,
                sent_at,
                sent_minute,
                date_source,
                occurrence,
                observed,
                observed,
            ),
        )
        if cur.rowcount:
            conn.execute(
                "INSERT INTO messages_fts (rowid, body) VALUES (?, ?)", (cur.lastrowid, body)
            )
            result.messages_added += 1
        else:
            conn.execute(
                "UPDATE messages SET last_observed_at = max(last_observed_at, ?) WHERE id = ?",
                (observed, mid),
            )
        result.message_ids[msg.get("label") or mid] = mid

    conn.commit()
    return result


def conversations(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM conversations ORDER BY rowid").fetchall()


def messages_in(conn: sqlite3.Connection, conv_id: str) -> list[sqlite3.Row]:
    """Messages of one conversation in ingest order."""
    return conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY rowid", (conv_id,)
    ).fetchall()
