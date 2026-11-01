"""Build a search-only baseline database for one scenario.

    python -m baseline.build --scenario a --db runs/p1/a.db

Loads the scenario, asks the LLM once per conversation for append-only
memories (each citing one message), and embeds every memory.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path
from typing import Any

from baseline.embed import Embedder, pack
from sme.llm.client import LLMClient
from sme.store import repo

BASELINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    message_id TEXT NOT NULL REFERENCES messages(id),
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id),
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);
"""

MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message_ref": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["message_ref", "text"],
            },
        }
    },
    "required": ["memories"],
}

PROMPT = """You write memories from one conversation so they can be found later by search.

Rules:
- Each memory is one short, standalone sentence that names who said or wrote it.
- Each memory cites exactly one message by its ref.
- Use only what the cited message says. Include its send date and time when it is known.
- Write as many memories as the messages need. Skip messages with nothing worth keeping.

Conversation: {title} ({app})
Messages, all times Asia/Kolkata:
{lines}
"""

APP_NAMES = {"whatsapp_web": "WhatsApp", "gmail": "Gmail"}


def init_baseline_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(BASELINE_SCHEMA)


def build_prompt(conv: sqlite3.Row, messages: list[sqlite3.Row]) -> str:
    lines = []
    for ref, msg in enumerate(messages, start=1):
        when = msg["sent_at"] or "time unknown"
        lines.append(f"[{ref}] {when} | {msg['sender_label']}: {msg['body']}")
    return PROMPT.format(
        title=conv["title"], app=APP_NAMES.get(conv["app"], conv["app"]), lines="\n".join(lines)
    )


def memory_id(conv_id: str, msg_id: str, position: int, text: str) -> str:
    return hashlib.sha256(f"{conv_id}|{msg_id}|{position}|{text}".encode()).hexdigest()


def extract_memories(conn: sqlite3.Connection, llm: LLMClient) -> list[tuple[str, str, str, str]]:
    """One LLM call per conversation. Returns (id, conversation_id, message_id, text) rows."""
    rows: list[tuple[str, str, str, str]] = []
    for conv in repo.conversations(conn):
        messages = repo.messages_in(conn, conv["id"])
        if not messages:
            continue
        refs = {str(i): m["id"] for i, m in enumerate(messages, start=1)}
        reply = llm.generate_json(build_prompt(conv, messages), MEMORY_SCHEMA)
        for position, item in enumerate(reply.get("memories", [])):
            ref = str(item.get("message_ref", "")).strip().strip("[]")
            text = str(item.get("text", "")).strip()
            if ref not in refs or not text:
                print(f"skipping memory with bad ref or empty text: {item!r}", file=sys.stderr)
                continue
            msg_id = refs[ref]
            rows.append((memory_id(conv["id"], msg_id, position, text), conv["id"], msg_id, text))
    return rows


def build(db_path: str | Path, scenario: dict[str, Any], llm: LLMClient, embedder: Embedder) -> int:
    """Build the baseline database. Returns the number of memories written."""
    conn = repo.connect(db_path)
    try:
        repo.init_schema(conn)
        init_baseline_schema(conn)
        if conn.execute("SELECT count(*) FROM memories").fetchone()[0]:
            raise RuntimeError(f"{db_path} already has memories; use --rebuild to start over")
        repo.load_scenario(conn, scenario)

        rows = extract_memories(conn, llm)
        vectors = embedder.embed([r[3] for r in rows]) if rows else []
        conn.executemany(
            "INSERT INTO memories (id, conversation_id, message_id, text) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.executemany(
            "INSERT INTO memory_vectors (memory_id, model, dim, vector) VALUES (?, ?, ?, ?)",
            [(r[0], embedder.name, len(v), pack(v)) for r, v in zip(rows, vectors, strict=True)],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", required=True, help="scenario name (a, b) or JSON path")
    parser.add_argument("--db", required=True, help="output SQLite path")
    parser.add_argument("--rebuild", action="store_true", help="delete the database first")
    args = parser.parse_args(argv)

    from baseline.embed import SentenceTransformerEmbedder
    from sme.llm.client import GeminiClient

    scenario = repo.read_scenario(args.scenario)
    llm = GeminiClient()
    embedder = SentenceTransformerEmbedder()

    db = Path(args.db)
    if args.rebuild and db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)
    count = build(db, scenario, llm, embedder)
    print(f"scenario {scenario['scenario']}: wrote {count} memories to {db}")


if __name__ == "__main__":
    main()
