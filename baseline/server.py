"""Search-only baseline MCP server over stdio.

    python -m baseline.server --db runs/p1/a.db --now 2026-11-12T20:00:00+05:30

Exposes exactly three tools. Every response carries `as_of` set to --now.
No tool decides what is current or what is owed; that is left to the model.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from baseline.embed import Embedder, cosine, unpack
from sme.store import repo

APP_NAMES = {"whatsapp_web": "WhatsApp", "gmail": "Gmail"}


def _newest_first(sent_at: str | None) -> tuple[int, float]:
    """Newest first, unknown time last."""
    if sent_at is None:
        return (1, 0.0)
    return (0, -repo.parse_time(sent_at).timestamp())


def _chronological(row: sqlite3.Row) -> tuple[int, float]:
    """Oldest first, unknown time last."""
    if row["sent_at"] is None:
        return (1, 0.0)
    return (0, repo.parse_time(row["sent_at"]).timestamp())


def _message(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "sender": row["sender_label"],
        "direction": row["direction"],
        "time": row["sent_at"],
        "date_source": row["date_source"],
        "text": row["body"],
    }


class BaselineTools:
    """Tool logic. Opens a connection per call because MCP runs sync tools in worker threads."""

    def __init__(self, db_path: str | Path, now: str, embedder: Embedder) -> None:
        self.db_path = db_path
        self.now = repo.parse_time(now)
        self.as_of = repo.to_iso(self.now)
        self.embedder = embedder

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        conn = repo.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def search_memory(self, query: str, limit: int = 5) -> dict[str, Any]:
        with self._db() as conn:
            rows = conn.execute(
                """SELECT mem.text AS memory, v.vector, c.app, c.title, m.sender_label,
                          m.direction, m.sent_at, m.date_source, m.body
                   FROM memories mem
                   JOIN memory_vectors v ON v.memory_id = mem.id
                   JOIN messages m ON m.id = mem.message_id
                   JOIN conversations c ON c.id = mem.conversation_id"""
            ).fetchall()
        results: list[dict[str, Any]] = []
        if rows and limit > 0:
            qvec = self.embedder.embed([query])[0]
            scored = sorted(
                ((cosine(qvec, unpack(r["vector"])), r) for r in rows),
                key=lambda pair: pair[0],
                reverse=True,
            )
            for score, r in scored[:limit]:
                results.append(
                    {
                        "memory": r["memory"],
                        "score": round(score, 4),
                        "source_app": APP_NAMES.get(r["app"], r["app"]),
                        "thread_title": r["title"],
                        "time": r["sent_at"],
                        "message": _message(r),
                    }
                )
        return {"as_of": self.as_of, "query": query, "results": results}

    def _threads(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        threads = []
        for conv in repo.conversations(conn):
            known = conn.execute(
                "SELECT sent_at FROM messages WHERE conversation_id = ? AND sent_at IS NOT NULL",
                (conv["id"],),
            ).fetchall()
            latest = max((r["sent_at"] for r in known), key=repo.parse_time, default=None)
            count = conn.execute(
                "SELECT count(*) FROM messages WHERE conversation_id = ?", (conv["id"],)
            ).fetchone()[0]
            threads.append(
                {
                    "id": conv["id"],
                    "source_app": APP_NAMES.get(conv["app"], conv["app"]),
                    "thread_title": conv["title"],
                    "latest_message_time": latest,
                    "message_count": count,
                }
            )
        threads.sort(key=lambda t: _newest_first(t["latest_message_time"]))
        return threads

    def list_active_threads(self) -> dict[str, Any]:
        with self._db() as conn:
            threads = [{k: v for k, v in t.items() if k != "id"} for t in self._threads(conn)]
        return {"as_of": self.as_of, "threads": threads}

    def get_latest_context(self, hours: float = 48) -> dict[str, Any]:
        since = self.now - timedelta(hours=hours)
        out = []
        with self._db() as conn:
            for t in self._threads(conn):
                latest = t["latest_message_time"]
                if latest is None or not since <= repo.parse_time(latest) <= self.now:
                    continue
                messages = sorted(
                    (
                        m
                        for m in repo.messages_in(conn, t["id"])
                        if m["sent_at"] is not None
                        and since <= repo.parse_time(m["sent_at"]) <= self.now
                    ),
                    key=_chronological,
                )
                out.append(
                    {
                        "source_app": t["source_app"],
                        "thread_title": t["thread_title"],
                        "latest_message_time": latest,
                        "messages": [_message(m) for m in messages],
                    }
                )
        return {
            "as_of": self.as_of,
            "window_hours": hours,
            "since": repo.to_iso(since),
            "threads": out,
        }


def build_server(tools: BaselineTools):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("sme-baseline")

    @server.tool()
    def search_memory(query: str, limit: int = 5) -> dict[str, Any]:
        """Search stored memories by meaning. Returns each memory with its source app,
        thread title, message time and the raw message with its sender."""
        return tools.search_memory(query, limit)

    @server.tool()
    def list_active_threads() -> dict[str, Any]:
        """List conversation threads with their latest message time, newest first.
        Threads with no known message time come last."""
        return tools.list_active_threads()

    @server.tool()
    def get_latest_context(hours: float = 48) -> dict[str, Any]:
        """Return every message sent in the last `hours` hours before as_of, grouped by
        thread. Messages with no known time are left out."""
        return tools.get_latest_context(hours)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search-only baseline MCP server (stdio)")
    parser.add_argument("--db", required=True, help="database built by baseline.build")
    parser.add_argument(
        "--now", required=True, help="ISO time with offset, e.g. 2026-11-12T20:00:00+05:30"
    )
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        parser.error(f"database not found: {args.db}")

    from baseline.embed import SentenceTransformerEmbedder

    tools = BaselineTools(args.db, args.now, SentenceTransformerEmbedder())
    build_server(tools).run("stdio")


if __name__ == "__main__":
    main()
