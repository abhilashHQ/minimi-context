"""Build a search-only baseline database for one scenario.

    python -m baseline.build --scenario a --db runs/p1/a.db

Loads the scenario, asks the LLM for exactly one memory per message, citing
that message, and embeds every memory. Conversations go to the LLM in chunks
of at most 40 messages, each with the 5 messages before it as context only.
Every finished chunk is committed with its memories, so a rerun after a crash
skips the chunks already done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from baseline.embed import Embedder, pack
from sme.llm.client import LLMClient
from sme.store import repo

CHUNK_SIZE = 40
CONTEXT_SIZE = 5
# Extra calls for messages the model left without a memory.
MISSING_RETRIES = 2

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
CREATE TABLE IF NOT EXISTS build_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS build_chunks (
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    chunk INTEGER NOT NULL,
    first_message_id TEXT NOT NULL REFERENCES messages(id),
    message_count INTEGER NOT NULL,
    PRIMARY KEY (conversation_id, chunk)
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
- Write exactly one memory for each numbered message. Write none for context lines.

Conversation: {title} ({app})
{context}Messages, all times Asia/Kolkata:
{lines}
"""

CONTEXT_BLOCK = """Earlier messages, for context only. They have no ref; write no memory for them:
{lines}

"""

APP_NAMES = {"whatsapp_web": "WhatsApp", "gmail": "Gmail"}

Log = Callable[[str], None]


def _print(line: str) -> None:
    print(line, flush=True)


def init_baseline_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(BASELINE_SCHEMA)


def _line(msg: sqlite3.Row) -> str:
    return f"{msg['sent_at'] or 'time unknown'} | {msg['sender_label']}: {msg['body']}"


def build_prompt(
    conv: sqlite3.Row, messages: list[sqlite3.Row], context: list[sqlite3.Row] | None = None
) -> str:
    lines = [f"[{ref}] {_line(msg)}" for ref, msg in enumerate(messages, start=1)]
    context_text = ""
    if context:
        context_text = CONTEXT_BLOCK.format(
            lines="\n".join(f"(context) {_line(m)}" for m in context)
        )
    return PROMPT.format(
        title=conv["title"],
        app=APP_NAMES.get(conv["app"], conv["app"]),
        context=context_text,
        lines="\n".join(lines),
    )


def memory_id(conv_id: str, msg_id: str, position: int, text: str) -> str:
    return hashlib.sha256(f"{conv_id}|{msg_id}|{position}|{text}".encode()).hexdigest()


def fallback_memory(conv: sqlite3.Row, msg: sqlite3.Row) -> str:
    """Used only when the model keeps skipping a message: the message, stated plainly."""
    app = APP_NAMES.get(conv["app"], conv["app"])
    when = msg["sent_at"] or "an unknown time"
    return f"{msg['sender_label']} wrote in {app} {conv['title']} at {when}: {msg['body']}"


def chunk_memories(
    llm: LLMClient,
    conv: sqlite3.Row,
    chunk: list[sqlite3.Row],
    context: list[sqlite3.Row],
    log: Log = _print,
) -> list[tuple[str, str]]:
    """Exactly one (message_id, text) per message of the chunk, in message order."""
    texts: dict[str, str] = {}
    targets, ctx = chunk, context
    for attempt in range(1 + MISSING_RETRIES):
        refs = {str(i): m["id"] for i, m in enumerate(targets, start=1)}
        reply = llm.generate_json(build_prompt(conv, targets, ctx), MEMORY_SCHEMA)
        items = reply.get("memories", []) if isinstance(reply, dict) else []
        for item in items:
            ref = str(item.get("message_ref", "")).strip().strip("[]")
            text = str(item.get("text", "")).strip()
            if ref not in refs or not text:
                print(f"skipping memory with bad ref or empty text: {item!r}", file=sys.stderr)
                continue
            texts.setdefault(refs[ref], text)
        missing = [m for m in chunk if m["id"] not in texts]
        if not missing:
            break
        if attempt < MISSING_RETRIES:
            log(f"  {len(missing)} messages got no memory, asking again")
            pos = chunk.index(missing[0])
            ctx = (context + chunk[:pos])[-CONTEXT_SIZE:]
            targets = missing
    for msg in chunk:
        if msg["id"] not in texts:
            print(
                f"no memory from the model for message {msg['id']}; using its text", file=sys.stderr
            )
            texts[msg["id"]] = fallback_memory(conv, msg)
    return [(m["id"], texts[m["id"]]) for m in chunk]


def scenario_fingerprint(scenario: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()


def _check_resumable(conn: sqlite3.Connection, db_path: str | Path, fingerprint: str) -> None:
    row = conn.execute("SELECT value FROM build_meta WHERE key = 'scenario'").fetchone()
    if row is None:
        if conn.execute("SELECT count(*) FROM memories").fetchone()[0]:
            raise RuntimeError(f"{db_path} already has memories; use --rebuild to start over")
        conn.execute("INSERT INTO build_meta (key, value) VALUES ('scenario', ?)", (fingerprint,))
        conn.commit()
    elif row["value"] != fingerprint:
        raise RuntimeError(
            f"{db_path} was built from a different scenario; use --rebuild to start over"
        )


def build(
    db_path: str | Path,
    scenario: dict[str, Any],
    llm: LLMClient,
    embedder: Embedder,
    chunk_size: int = CHUNK_SIZE,
    context_size: int = CONTEXT_SIZE,
    log: Log = _print,
) -> int:
    """Build or resume the baseline database. Returns the memories written by this call."""
    conn = repo.connect(db_path)
    try:
        repo.init_schema(conn)
        init_baseline_schema(conn)
        _check_resumable(conn, db_path, scenario_fingerprint(scenario))
        repo.load_scenario(conn, scenario)

        plan = []
        for conv in repo.conversations(conn):
            messages = repo.messages_in(conn, conv["id"])
            for index, start in enumerate(range(0, len(messages), chunk_size)):
                plan.append((conv, index, start, messages))
        done = {
            (r["conversation_id"], r["chunk"])
            for r in conn.execute("SELECT conversation_id, chunk FROM build_chunks")
        }
        total_chunks = len(plan)
        skipped = sum(1 for conv, index, _, _ in plan if (conv["id"], index) in done)
        if skipped:
            log(f"resuming: {skipped}/{total_chunks} chunks already done")

        written = 0
        for n, (conv, index, start, messages) in enumerate(plan, start=1):
            if (conv["id"], index) in done:
                continue
            chunk = messages[start : start + chunk_size]
            context = messages[max(0, start - context_size) : start]
            pairs = chunk_memories(llm, conv, chunk, context, log)
            rows = [
                (memory_id(conv["id"], mid, 0, text), conv["id"], mid, text) for mid, text in pairs
            ]
            vectors = embedder.embed([r[3] for r in rows])
            with conn:
                conn.executemany(
                    "INSERT INTO memories (id, conversation_id, message_id, text)"
                    " VALUES (?, ?, ?, ?)",
                    rows,
                )
                conn.executemany(
                    "INSERT INTO memory_vectors (memory_id, model, dim, vector)"
                    " VALUES (?, ?, ?, ?)",
                    [
                        (r[0], embedder.name, len(v), pack(v))
                        for r, v in zip(rows, vectors, strict=True)
                    ],
                )
                conn.execute(
                    "INSERT INTO build_chunks (conversation_id, chunk, first_message_id,"
                    " message_count) VALUES (?, ?, ?, ?)",
                    (conv["id"], index, chunk[0]["id"], len(chunk)),
                )
            written += len(rows)
            log(
                f"[{n}/{total_chunks}] {conv['title']} chunk {index + 1}:"
                f" {len(chunk)} messages, {len(rows)} memories"
            )
        return written
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", required=True, help="scenario name (a, b, c) or JSON path")
    parser.add_argument("--db", required=True, help="output SQLite path")
    parser.add_argument(
        "--rebuild", action="store_true", help="delete the database first instead of resuming"
    )
    parser.add_argument("--llm", choices=["gemini", "bedrock"], default="gemini")
    args = parser.parse_args(argv)

    from baseline.embed import SentenceTransformerEmbedder
    from sme.llm.client import BedrockClient, GeminiClient

    scenario = repo.read_scenario(args.scenario)
    llm = BedrockClient() if args.llm == "bedrock" else GeminiClient()
    embedder = SentenceTransformerEmbedder()

    db = Path(args.db)
    if args.rebuild and db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)
    count = build(db, scenario, llm, embedder)
    conn = repo.connect(db)
    total = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    conn.close()
    print(f"scenario {scenario['scenario']}: wrote {count} memories ({total} in total) to {db}")


if __name__ == "__main__":
    main()
