"""W1 to W3: store raw sources, gate bulk mail, and group sources into bursts.

The burst rule is the same live and in replay: a burst closes after 60 minutes of quiet,
4 hours after its first message, or at 20 messages. Quiet is measured on the arrival clock.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from harness.store import Store
from harness.text import dt, is_you

QUIET = timedelta(minutes=60)
SPAN = timedelta(hours=4)
MAX_MESSAGES = 20
BULK_LABELS = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_UPDATES", "CATEGORY_FORUMS"}
MAIL_APPS = {"gmail", "gmail_connector"}


def _h(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def to_source(msg: dict[str, Any], titles: dict[str, str]) -> dict[str, Any]:
    """Map a corpus message to a source row. The content hash is what W1 deduplicates on: an
    email is one email whichever path (screen or connector) brought it in."""
    app = msg["app"]
    kind = "calendar" if app == "calendar" else "gmail" if app == "gmail_connector" else "screen"
    sender = msg["sender"]
    said = msg["sent_at"]
    if app in MAIL_APPS:
        content_hash = _h("mail", msg["thread"], _norm(sender), said[:16])
    elif app == "calendar":
        content_hash = _h("cal", msg["thread"], said, _norm(msg["body"]))
    else:
        content_hash = _h("chat", msg["thread"], _norm(sender), said, _norm(msg["body"]))
    event = msg.get("event")
    participants = [p for p in (event or {}).get("attendees", []) if not is_you(p)]
    extra = {k: msg[k] for k in ("labels", "sender_email", "event", "source") if k in msg}
    return {
        "label": msg.get("label"),
        "kind": kind,
        "app": app,
        "external_id": msg.get("external_id"),
        "thread_key": msg["thread"],
        "thread_title": titles.get(msg["thread"], msg["thread"]),
        "author": sender,
        "participants": participants,
        "said_at": said,
        "seen_at": msg.get("seen_at") or said,
        "text": msg["body"],
        "content_hash": content_hash,
        "extra": extra,
    }


def gate_reason(store: Store, src: dict[str, Any]) -> str | None:
    """W2: promo, social or bulk mail from a sender the user never replied to makes no memory."""
    if src["app"] not in MAIL_APPS:
        return None
    labels = set(src["extra"].get("labels", []))
    email = (src["extra"].get("sender_email") or "").lower()
    bulk = (
        bool(labels & BULK_LABELS)
        or email.startswith(("noreply", "no-reply", "donotreply", "newsletter", "offers", "deals"))
        or "unsubscribe" in src["text"].lower()
    )
    if not bulk:
        return None
    replied = store.db.execute(
        "SELECT 1 FROM sources WHERE thread_key = ? AND lower(author) IN ('you', 'me') LIMIT 1",
        (src["thread_key"],),
    ).fetchone()
    return None if replied else "bulk mail, never replied"


class Bursts:
    def __init__(self, store: Store) -> None:
        self.store = store

    def _open(self, thread_key: str):
        return self.store.db.execute(
            "SELECT * FROM bursts WHERE thread_key = ? AND status = 'open'", (thread_key,)
        ).fetchone()

    def _close(self, burst_id: int, at: str) -> None:
        self.store.db.execute(
            "UPDATE bursts SET status = 'ready', closed_at = ? WHERE id = ?", (at, burst_id)
        )

    def tick(self, now: str) -> None:
        """Close every open burst that has been quiet for QUIET on the arrival clock."""
        cutoff = dt(now) - QUIET
        for b in self.store.db.execute("SELECT * FROM bursts WHERE status = 'open'").fetchall():
            if dt(b["last_seen_at"]) <= cutoff:
                self._close(b["id"], now)

    def add(self, sid: int, src: dict[str, Any]) -> int:
        said, seen = dt(src["said_at"]), src["seen_at"]
        b = self._open(src["thread_key"])
        if b is not None and (
            said - dt(b["last_said_at"]) > QUIET
            or said - dt(b["first_said_at"]) > SPAN
            or b["n"] >= MAX_MESSAGES
        ):
            self._close(b["id"], seen)
            b = None
        if b is None:
            cur = self.store.db.execute(
                "INSERT INTO bursts(thread_key, first_said_at, last_said_at, last_seen_at, n)"
                " VALUES (?, ?, ?, ?, 0)",
                (src["thread_key"], src["said_at"], src["said_at"], seen),
            )
            bid = cur.lastrowid
        else:
            bid = b["id"]
        self.store.db.execute(
            "UPDATE bursts SET n = n + 1, last_said_at = max(last_said_at, ?),"
            " first_said_at = min(first_said_at, ?), last_seen_at = max(last_seen_at, ?) WHERE id = ?",
            (src["said_at"], src["said_at"], seen, bid),
        )
        self.store.db.execute("UPDATE sources SET burst_id = ? WHERE id = ?", (bid, sid))
        return bid

    def close_all(self, at: str) -> None:
        for b in self.store.db.execute("SELECT id FROM bursts WHERE status = 'open'").fetchall():
            self._close(b["id"], at)

    def ready(self) -> list[int]:
        rows = self.store.db.execute(
            "SELECT id FROM bursts WHERE status = 'ready' ORDER BY closed_at, last_said_at, id"
        )
        return [r[0] for r in rows]


def ingest(store: Store, bursts: Bursts, src: dict[str, Any]) -> tuple[int | None, str]:
    """W1 and W2 and W3 for one source. Returns (source id or None, outcome)."""
    reason = gate_reason(store, src)
    if reason:
        src = {**src, "skip_reason": reason}
    sid = store.add_source(src)
    if sid is None:
        return None, "duplicate"
    if reason:
        return sid, "gated"
    bursts.add(sid, src)
    return sid, "added"


def load_corpus(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrival_order(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """Real-life order: by when minimi saw each message, then when it was said."""
    return sorted(corpus["messages"], key=lambda m: (m.get("seen_at") or m["sent_at"], m["sent_at"]))
