"""Append new messages to a built baseline database, one memory per message, the same way
baseline.build writes them. Used for the live moment so both systems see the same new message."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from baseline.build import CONTEXT_SIZE, chunk_memories, memory_id
from baseline.embed import Embedder, pack
from sme.llm.client import LLMClient
from sme.store import repo


def add_messages(db_path: str | Path, app: str, thread: str, title: str, messages: list[dict[str, Any]],
                 llm: LLMClient, embedder: Embedder) -> list[str]:
    conn = repo.connect(db_path)
    try:
        before = {r["id"] for r in conn.execute("SELECT id FROM messages")}
        repo.load_scenario(conn, {
            "scenario": "live", "now": messages[-1]["sent_at"],
            "conversations": [{"app": app, "key": thread, "title": title}],
            "messages": messages,
        })
        conv = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (repo.conversation_id(app, thread),)
        ).fetchone()
        rows = repo.messages_in(conn, conv["id"])
        new = [m for m in rows if m["id"] not in before]
        if not new:
            return []
        start = rows.index(new[0])
        pairs = chunk_memories(llm, conv, new, rows[max(0, start - CONTEXT_SIZE) : start], log=lambda _: None)
        mem_rows = [(memory_id(conv["id"], mid, 0, text), conv["id"], mid, text) for mid, text in pairs]
        vectors = embedder.embed([r[3] for r in mem_rows])
        with conn:
            conn.executemany(
                "INSERT INTO memories (id, conversation_id, message_id, text) VALUES (?, ?, ?, ?)", mem_rows
            )
            conn.executemany(
                "INSERT INTO memory_vectors (memory_id, model, dim, vector) VALUES (?, ?, ?, ?)",
                [(r[0], embedder.name, len(v), pack(v)) for r, v in zip(mem_rows, vectors, strict=True)],
            )
        return [r[3] for r in mem_rows]
    finally:
        conn.close()
