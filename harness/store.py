"""SQLite store for the memory harness: raw sources, bursts, memories, links and the op log.

Nothing is deleted. A memory is current while valid_to is null and it hasn't expired.
Vectors live in a BLOB column and in an in-process numpy matrix rebuilt on open.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

OPEN_KINDS = ("todo", "event", "status")
KINDS = ("event", "todo", "fact", "preference", "status", "episode")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    label TEXT,
    kind TEXT NOT NULL,
    app TEXT NOT NULL,
    external_id TEXT,
    thread_key TEXT NOT NULL,
    thread_title TEXT,
    author TEXT NOT NULL,
    participants TEXT NOT NULL DEFAULT '[]',
    said_at TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    skip_reason TEXT,
    burst_id INTEGER,
    extra TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS sources_thread ON sources(thread_key, said_at);
CREATE TABLE IF NOT EXISTS bursts (
    id INTEGER PRIMARY KEY,
    thread_key TEXT NOT NULL,
    first_said_at TEXT NOT NULL,
    last_said_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    n INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    attempts INTEGER NOT NULL DEFAULT 0,
    closed_at TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    kind TEXT NOT NULL,
    owner TEXT,
    people TEXT NOT NULL DEFAULT '[]',
    event_at TEXT,
    event_end TEXT,
    expires_at TEXT,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    end_reason TEXT,
    source_count INTEGER NOT NULL DEFAULT 1,
    created_burst INTEGER,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    quote TEXT,
    role TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ms_memory ON memory_sources(memory_id);
CREATE INDEX IF NOT EXISTS ms_source ON memory_sources(source_id);
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY,
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    made_by TEXT NOT NULL,
    at TEXT NOT NULL,
    source_id INTEGER,
    reversed_at TEXT
);
CREATE TABLE IF NOT EXISTS ops_log (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    burst_id INTEGER,
    stage TEXT NOT NULL,
    op TEXT,
    target_id INTEGER,
    memory_id INTEGER,
    outcome TEXT NOT NULL,
    reason TEXT,
    raw_json TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    tokens_in INTEGER,
    tokens_out INTEGER
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, tokenize='porter');
CREATE VIRTUAL TABLE IF NOT EXISTS sources_fts USING fts5(text, tokenize='porter');
"""

MEMORY_COLS = (
    "id, text, kind, owner, people, event_at, event_end, expires_at, valid_from, valid_to,"
    " end_reason, source_count, created_burst"
)
FTS_WORD = re.compile(r"[A-Za-z0-9]+")
FTS_STOP = set(
    "a an the to of and or for in on at by is are i you your me my we it this that be with"
    " can will so do did does what when who how any have has had there their about get got"
    " still just all some was were been am im i'm its it's whats what's anything".split()
)


def fts_query(text: str) -> str | None:
    words = [w.lower() for w in FTS_WORD.findall(text)]
    words = [w for w in dict.fromkeys(words) if w not in FTS_STOP and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in words) or None


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()
        self._load_vectors()

    def _load_vectors(self) -> None:
        rows = self.db.execute("SELECT id, embedding FROM memories ORDER BY id").fetchall()
        self.vec_ids = [r["id"] for r in rows]
        self.vec_pos = {mid: i for i, mid in enumerate(self.vec_ids)}
        self.vecs = (
            np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 384), dtype=np.float32)
        )
        self.loaded_version = self.data_version()

    def data_version(self) -> int:
        return self.db.execute("PRAGMA data_version").fetchone()[0]

    def refresh(self) -> None:
        """Reload vectors when another process (the CLI) has written since we last looked."""
        if self.data_version() != self.loaded_version:
            self._load_vectors()

    def commit(self) -> None:
        self.db.commit()

    # meta

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # sources

    def add_source(self, src: dict[str, Any]) -> int | None:
        """Insert a source. Returns None when its content hash is already stored (W1)."""
        try:
            cur = self.db.execute(
                "INSERT INTO sources(label, kind, app, external_id, thread_key, thread_title, author,"
                " participants, said_at, seen_at, text, content_hash, skip_reason, extra)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    src.get("label"), src["kind"], src["app"], src.get("external_id"),
                    src["thread_key"], src.get("thread_title"), src["author"],
                    json.dumps(src.get("participants", [])), src["said_at"], src["seen_at"],
                    src["text"], src["content_hash"], src.get("skip_reason"),
                    json.dumps(src.get("extra", {})),
                ),
            )
        except sqlite3.IntegrityError:
            return None
        self.db.execute(
            "INSERT INTO sources_fts(rowid, text) VALUES (?, ?)", (cur.lastrowid, src["text"])
        )
        return cur.lastrowid

    def source(self, sid: int) -> dict[str, Any]:
        return self._source_row(self.db.execute("SELECT * FROM sources WHERE id = ?", (sid,)).fetchone())

    def sources(self, where: str = "1", params: tuple = ()) -> list[dict[str, Any]]:
        rows = self.db.execute(f"SELECT * FROM sources WHERE {where} ORDER BY said_at, id", params)
        return [self._source_row(r) for r in rows]

    @staticmethod
    def _source_row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["participants"] = json.loads(d["participants"])
        d["extra"] = json.loads(d["extra"])
        return d

    # memories

    def add_memory(self, m: dict[str, Any], embedding: np.ndarray) -> int:
        emb = np.asarray(embedding, dtype=np.float32)
        cur = self.db.execute(
            "INSERT INTO memories(text, kind, owner, people, event_at, event_end, expires_at,"
            " valid_from, valid_to, end_reason, source_count, created_burst, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                m["text"], m["kind"], m.get("owner"), json.dumps(m.get("people", [])),
                m.get("event_at"), m.get("event_end"), m.get("expires_at"), m["valid_from"],
                m.get("valid_to"), m.get("end_reason"), 1, m.get("created_burst"), emb.tobytes(),
            ),
        )
        mid = cur.lastrowid
        self.db.execute("INSERT INTO memories_fts(rowid, text) VALUES (?, ?)", (mid, m["text"]))
        self.vec_pos[mid] = len(self.vec_ids)
        self.vec_ids.append(mid)
        self.vecs = np.vstack([self.vecs, emb[None, :]])
        return mid

    def memory(self, mid: int) -> dict[str, Any] | None:
        r = self.db.execute(f"SELECT {MEMORY_COLS} FROM memories WHERE id = ?", (mid,)).fetchone()
        return self._memory_row(r) if r else None

    def memories(self, where: str = "1", params: tuple = ()) -> list[dict[str, Any]]:
        rows = self.db.execute(f"SELECT {MEMORY_COLS} FROM memories WHERE {where} ORDER BY id", params)
        return [self._memory_row(r) for r in rows]

    def current(self, at: str | None = None) -> list[dict[str, Any]]:
        if at is None:
            return self.memories("valid_to IS NULL")
        return self.memories(
            "valid_to IS NULL AND (expires_at IS NULL OR expires_at > ?)", (at,)
        )

    @staticmethod
    def _memory_row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["people"] = json.loads(d["people"])
        return d

    def end_memory(self, mid: int, at: str, reason: str) -> None:
        self.db.execute(
            "UPDATE memories SET valid_to = ?, end_reason = ? WHERE id = ? AND valid_to IS NULL",
            (at, reason, mid),
        )

    def attach_source(self, mid: int, sid: int, quote: str | None, role: str) -> None:
        self.db.execute(
            "INSERT INTO memory_sources(memory_id, source_id, quote, role) VALUES (?, ?, ?, ?)",
            (mid, sid, quote, role),
        )
        if role != "created":
            self.db.execute("UPDATE memories SET source_count = source_count + 1 WHERE id = ?", (mid,))

    def memory_sources(self, mid: int) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT ms.quote, ms.role, s.* FROM memory_sources ms JOIN sources s ON s.id = ms.source_id"
            " WHERE ms.memory_id = ? ORDER BY s.said_at",
            (mid,),
        )
        return [self._source_row(r) for r in rows]

    def thread_memory_ids(self, thread_key: str) -> list[int]:
        rows = self.db.execute(
            "SELECT DISTINCT ms.memory_id FROM memory_sources ms JOIN sources s ON s.id = ms.source_id"
            " WHERE s.thread_key = ?",
            (thread_key,),
        )
        return [r[0] for r in rows]

    def link(self, from_id: int, to_id: int, type_: str, made_by: str, at: str, sid: int | None) -> None:
        self.db.execute(
            "INSERT INTO links(from_id, to_id, type, made_by, at, source_id) VALUES (?, ?, ?, ?, ?, ?)",
            (from_id, to_id, type_, made_by, at, sid),
        )

    def links_to(self, mid: int) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM links WHERE to_id = ? AND reversed_at IS NULL ORDER BY id", (mid,)
        )
        return [dict(r) for r in rows]

    def links_from(self, mid: int) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM links WHERE from_id = ? AND reversed_at IS NULL ORDER BY id", (mid,)
        )
        return [dict(r) for r in rows]

    def log(self, at: str, stage: str, outcome: str, *, burst_id: int | None = None,
            op: str | None = None, target_id: int | None = None, memory_id: int | None = None,
            reason: str | None = None, raw: Any = None) -> None:
        self.db.execute(
            "INSERT INTO ops_log(at, burst_id, stage, op, target_id, memory_id, outcome, reason, raw_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (at, burst_id, stage, op, target_id, memory_id, outcome, reason,
             json.dumps(raw, default=str) if raw is not None else None),
        )

    # search primitives

    def vector_top(self, query: np.ndarray, ids: list[int] | None, k: int) -> list[tuple[int, float]]:
        if not self.vec_ids:
            return []
        if ids is None:
            pos = np.arange(len(self.vec_ids))
        else:
            pos = np.array([self.vec_pos[i] for i in ids if i in self.vec_pos], dtype=int)
            if not len(pos):
                return []
        sims = self.vecs[pos] @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-sims)[:k]
        return [(self.vec_ids[pos[i]], float(sims[i])) for i in order]

    def vector(self, mid: int) -> np.ndarray:
        return self.vecs[self.vec_pos[mid]]

    def bm25_memories(self, text: str, k: int, allowed: set[int] | None = None) -> list[int]:
        q = fts_query(text)
        if not q:
            return []
        rows = self.db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?",
            (q, k * 10 if allowed is not None else k),
        )
        ids = [r[0] for r in rows]
        if allowed is not None:
            ids = [i for i in ids if i in allowed]
        return ids[:k]

    def bm25_sources(self, text: str, k: int, where: str = "1", params: tuple = ()) -> list[dict[str, Any]]:
        q = fts_query(text)
        if not q:
            return []
        rows = self.db.execute(
            f"SELECT s.* FROM sources_fts f JOIN sources s ON s.id = f.rowid"
            f" WHERE sources_fts MATCH ? AND {where} ORDER BY bm25(sources_fts) LIMIT ?",
            (q, *params, k),
        )
        return [self._source_row(r) for r in rows]
