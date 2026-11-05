"""Baseline retrieval behind the tool surface of minimi's hosted connector, over stdio.

    python -m baseline.minimi_shape --db runs/p1c/c.db --now 2026-11-26T20:00:00+05:30

Tool names, input fields, descriptions and server instructions follow what the hosted
connector returns from initialize and tools/list. Retrieval is the baseline's: cosine
search over memories plus recency listings, now with time, app and paging filters.
The corpus has no meetings, so meeting_memory finds none.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from baseline.embed import Embedder, cosine, unpack
from baseline.server import APP_NAMES, BaselineTools, _chronological, _message, _newest_first
from sme.store import repo

INSTRUCTIONS = """# Minimi connector

Minimi is a desktop app running on this user's laptop right now. It
continuously captures what the user is doing across their chat apps
(WhatsApp, Slack, Discord, Teams, iMessage), email (Apple Mail, Outlook,
Spark), documents (Notes, Obsidian, Notion, Word, Pages), tasks (Todoist,
OmniFocus, Microsoft To Do), calendar (macOS Calendar, Fantastical), and
anything else they read or type in a browser or native window. The capture
is stored locally on the user's laptop in an encrypted SQLite database and
is only retrievable through this connector while the laptop is online.

## What this means for you

When the user asks a question about THEIR OWN recent activity - what they
sent, received, wrote, read, looked at, clicked, or scheduled - minimi
is the ONLY source of ground truth. Your training data does not contain
this user's messages, emails, docs, or calendar. Do not guess, do not
describe a generic answer, and do not say 'I don't have access to your
messages' - you do, through these tools.

## Workflow policy

1. **Question about the user's own activity? Call a minimi tool
   first.** Trigger phrases: 'my', 'latest', 'last', 'just now',
   'today', 'yesterday', 'this week', 'on my screen', 'what did I',
   'what did <person> say', 'what's in my inbox', 'what am I working
   on'. See each tool's own description for what it does.

2. **Orient before you drill in.** If the question is vague or broad
   ('what was I doing tonight', 'catch me up', 'what's going on'),
   call `list_active_threads` first (or `get_latest_context` for
   'right now'). Read what threads/people/apps are active. Then, if
   you still need deeper content, call `search_memory` with a
   SHARPENED query built from what you saw. Never feed vague user
   phrasing straight into `search_memory` - it ranks by topic and
   returns noise on broad queries. Same pattern inside
   `meeting_memory`: `list` to orient, then `search` with a concrete
   query, then `get_context` to drill in.

3. **If a minimi tool returns nothing relevant**, fall back to your
   own reasoning and say clearly that minimi had no record of it.

## Freshness & timezone

- Memory updates continuously. A tool response is a snapshot of this
  moment - if the user follows up with 'and now?', call the tool again.
  Do not reuse an earlier response to answer a later 'latest' question.
- Every tool response includes the user's local time (`as_of_local`) and
  IANA timezone (`tz`). Interpret 'today', 'this morning', 'last hour',
  etc. in that local frame, not UTC and not your own assumed timezone.

## When the device is offline

If a tool reports the device is not connected, tell the user plainly:
minimi needs their laptop online to answer. Do not fabricate an answer
from general knowledge in that case.
"""

SEARCH_DESC = (
    "Semantic search across the user's captured memory (chat, email, docs, tasks, calendar,"
    " web, etc.). Ranks by topic similarity to `query`. Pick this when you have a specific"
    " topic, person, or past event. Pick `get_latest_context` for 'now/latest/on my screen'"
    " (recency, not topic). Pick `list_active_threads` for broad/vague questions (it's also"
    " the orient step before coming here - see server instructions). Pick `meeting_memory`"
    " for any meeting/call content. Paginated via `cursor`."
)
THREADS_DESC = (
    "Index of recently-active threads (chats, docs, tasks, email) - metadata + short previews"
    " only, no full content, no semantic search. Pick this for overviews and for vague/broad"
    " questions (it's the orient step in the server-instructions workflow). Pick"
    " `search_memory` once you have a specific topic. Pick `get_latest_context` for full"
    " content of what's open RIGHT NOW. Pick `meeting_memory` for meetings/calls. Response"
    " includes `tz` + `as_of_local`."
)
LATEST_DESC = (
    "Full content of the most-recently-updated thread(s), newest first. Ranked by freshness"
    " only, no topic filter. Pick this for 'now/latest/on my screen/just now' questions. Pick"
    " `list_active_threads` for broader overviews or vague questions. Pick `search_memory`"
    " for a specific topic. Pick `meeting_memory` for meetings/calls. Response includes `tz`"
    " + `as_of_local`."
)
MEETING_DESC = (
    "Meetings, calls, huddles, and voice recordings (Zoom, Meet, Teams, Webex, Slack/Huddle,"
    " WhatsApp, Discord, Whereby, Gather, Around, FaceTime, Phone, Voice Note). Use for ANY"
    " meeting/call/voice-note question - do not use `search_memory` or `list_active_threads`"
    " for these. Actions: `list` (count/list in a time range, also the orient step for vague"
    " meeting questions - see server instructions), `search` (semantic search within"
    " transcripts, requires a specific query), `get_context` (full transcript by thread_id"
    " from a prior `list`/`search`). Response includes `tz` + `as_of_local`."
)
SEARCH_TIME_RANGE_DESC = """\
Restrict results to memories captured within this time range. Three usage shapes:
  1. Duration look-back: { last: { hours: 3 } }
  2. Calendar day(s) with optional time-of-day window:
       { day: 'yesterday' }
       { day: '2026-05-12', between: { from: '14:00', to: '15:00' } }
       { day_range: { from: '2026-05-01', to: '2026-05-05' } }
  3. (Legacy) Explicit epoch-ms window - pass numeric `from` and `to` fields.
     Prefer the calendar shapes above.
`last` is a pure look-back duration (minutes/hours/days, summed); do NOT use it with day/between.
`day` is 'today', 'yesterday', or 'YYYY-MM-DD' in the user's local timezone.
`day_range` is an inclusive { from, to } of 'YYYY-MM-DD' dates.
`between` is a local 'HH:MM' { from, to } window; pair with `day` or `day_range`,
alone it implies today."""


class TimeRangeError(ValueError):
    pass


def _day(value: str, now: datetime) -> date:
    if value == "today":
        return now.date()
    if value == "yesterday":
        return now.date() - timedelta(days=1)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise TimeRangeError(f"bad day {value!r}; use 'today', 'yesterday' or YYYY-MM-DD") from exc


def _clock(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise TimeRangeError(f"bad time {value!r}; use HH:MM") from exc


def resolve_time_range(
    spec: dict[str, Any] | None, now: datetime
) -> list[tuple[datetime, datetime]]:
    """Turn a connector time_range into [start, end] windows in the user's timezone.

    Returns an empty list when no range is given. A day range with `between` gives one
    window per day.
    """
    spec = {k: v for k, v in (spec or {}).items() if v is not None}
    if not spec:
        return []
    tz = now.tzinfo
    if "last" in spec:
        last = spec["last"] or {}
        delta = timedelta(
            minutes=last.get("minutes", 0), hours=last.get("hours", 0), days=last.get("days", 0)
        )
        if delta <= timedelta(0):
            raise TimeRangeError("`last` needs a positive minutes, hours or days")
        return [(now - delta, now)]
    if "day_range" in spec:
        first = _day(spec["day_range"]["from"], now)
        last_day = _day(spec["day_range"]["to"], now)
        days = [first + timedelta(days=i) for i in range((last_day - first).days + 1)]
    elif "day" in spec:
        days = [_day(spec["day"], now)]
    elif "between" in spec:
        days = [now.date()]
    elif "from" in spec and "to" in spec:
        return [
            (
                datetime.fromtimestamp(spec["from"] / 1000, tz),
                datetime.fromtimestamp(spec["to"] / 1000, tz),
            )
        ]
    else:
        raise TimeRangeError("time_range needs last, day, day_range, between or from/to")
    between = spec.get("between")
    start_t = _clock(between["from"]) if between else time.min
    end_t = _clock(between["to"]) if between else time.max
    return [(datetime.combine(d, start_t, tz), datetime.combine(d, end_t, tz)) for d in days]


def _window(name: str, now: datetime) -> tuple[datetime, datetime]:
    today = datetime.combine(now.date(), time.min, now.tzinfo)
    if name == "last_hour":
        return now - timedelta(hours=1), now
    if name == "yesterday":
        return today - timedelta(days=1), today
    if name == "this_week":
        return today - timedelta(days=now.weekday()), now
    return today, now


def _inside(sent_at: str | None, windows: list[tuple[datetime, datetime]]) -> bool:
    if not windows:
        return True
    if sent_at is None:
        return False
    t = repo.parse_time(sent_at)
    return any(start <= t <= end for start, end in windows)


def _app_matches(app: str, wanted: list[str] | None) -> bool:
    if not wanted:
        return True
    shown = APP_NAMES.get(app, app).lower()
    return shown in {w.lower() for w in wanted}


class MinimiShapeTools(BaselineTools):
    def __init__(self, db_path: str | Path, now: str, embedder: Embedder, tz: str) -> None:
        super().__init__(db_path, now, embedder)
        self.tz = tz

    def _stamp(self, out: dict[str, Any]) -> dict[str, Any]:
        return {"as_of": self.as_of, "as_of_local": self.as_of, "tz": self.tz, **out}

    def search(
        self,
        query: str,
        limit: int = 5,
        cursor: int = 0,
        time_range: dict[str, Any] | None = None,
        source_app: str | None = None,
    ) -> dict[str, Any]:
        windows = resolve_time_range(time_range, self.now)
        apps = [source_app] if source_app else None
        with self._db() as conn:
            rows = [
                r
                for r in conn.execute(
                    """SELECT mem.text AS memory, v.vector, c.app, c.title, m.sender_label,
                              m.direction, m.sent_at, m.date_source, m.body
                       FROM memories mem
                       JOIN memory_vectors v ON v.memory_id = mem.id
                       JOIN messages m ON m.id = mem.message_id
                       JOIN conversations c ON c.id = mem.conversation_id"""
                ).fetchall()
                if _inside(r["sent_at"], windows) and _app_matches(r["app"], apps)
            ]
        results: list[dict[str, Any]] = []
        next_cursor = None
        if rows:
            qvec = self.embedder.embed([query])[0]
            scored = sorted(
                ((cosine(qvec, unpack(r["vector"])), r) for r in rows),
                key=lambda pair: pair[0],
                reverse=True,
            )
            page = scored[cursor : cursor + limit]
            if cursor + limit < len(scored):
                next_cursor = cursor + limit
            for score, r in page:
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
        return self._stamp({"query": query, "results": results, "next_cursor": next_cursor})

    def active_threads(
        self,
        window: str = "today",
        time_range: dict[str, Any] | None = None,
        apps: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        windows = resolve_time_range(time_range, self.now) or [_window(window, self.now)]
        out = []
        with self._db() as conn:
            for t in self._threads(conn):
                conv = conn.execute(
                    "SELECT app FROM conversations WHERE id = ?", (t["id"],)
                ).fetchone()
                if not _app_matches(conv["app"], apps):
                    continue
                inside = sorted(
                    (m for m in repo.messages_in(conn, t["id"]) if _inside(m["sent_at"], windows)),
                    key=_chronological,
                )
                if not inside:
                    continue
                last = inside[-1]
                out.append(
                    {
                        "source_app": t["source_app"],
                        "thread_title": t["thread_title"],
                        "latest_message_time": last["sent_at"],
                        "message_count": len(inside),
                        "preview": f"{last['sender_label']}: {last['body'][:120]}",
                    }
                )
        out.sort(key=lambda t: _newest_first(t["latest_message_time"]))
        return self._stamp({"threads": out[:limit]})

    def latest_context(
        self, hours: float = 2, apps: list[str] | None = None, limit: int = 3
    ) -> dict[str, Any]:
        since = self.now - timedelta(hours=hours)
        out = []
        with self._db() as conn:
            for t in self._threads(conn):
                conv = conn.execute(
                    "SELECT app FROM conversations WHERE id = ?", (t["id"],)
                ).fetchone()
                latest = t["latest_message_time"]
                if not _app_matches(conv["app"], apps) or latest is None:
                    continue
                if not since <= repo.parse_time(latest) <= self.now:
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
        return self._stamp(
            {"window_hours": hours, "since": repo.to_iso(since), "threads": out[:limit]}
        )

    def meetings(self, action: str, thread_id: str | None = None) -> dict[str, Any]:
        if action == "get_context":
            return self._stamp({"error": f"no meeting with thread_id {thread_id!r}"})
        return self._stamp({"meetings": []})


def build_server(tools: MinimiShapeTools):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("minimi", title="Minimi", instructions=INSTRUCTIONS)

    def guarded(fn, *args, **kwargs) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except TimeRangeError as exc:
            return {"error": str(exc)}

    @server.tool(description=SEARCH_DESC)
    def search_memory(
        query: Annotated[
            str,
            Field(
                description="Concrete topic/person/project/event. Good: 'budget email from"
                " Priya', 'messages with Father about weekend plans', 'doc where I wrote the"
                " migration plan'. Bad: 'what was I doing', 'recent stuff', 'tonight'."
            ),
        ],
        limit: Annotated[
            int, Field(ge=1, le=20, description="Max number of results per page (default 5).")
        ] = 5,
        cursor: Annotated[
            int,
            Field(
                ge=0,
                description="Offset into the ranked result list. Omit (or pass 0) for the first"
                " page. Use the `cursor` value from a prior response to fetch the next page.",
            ),
        ] = 0,
        time_range: Annotated[
            dict[str, Any] | None, Field(description=SEARCH_TIME_RANGE_DESC)
        ] = None,
        source_app: Annotated[
            str | None, Field(description="Restrict to a specific app, e.g. 'WhatsApp'.")
        ] = None,
    ) -> dict[str, Any]:
        return guarded(tools.search, query, limit, cursor, time_range, source_app)

    @server.tool(description=THREADS_DESC)
    def list_active_threads(
        window: Annotated[
            Literal["last_hour", "today", "yesterday", "this_week"],
            Field(
                description="Convenience time window (default: today). Use instead of"
                " time_range for common cases."
            ),
        ] = "today",
        time_range: Annotated[
            dict[str, Any] | None,
            Field(
                description="Explicit time range { from: start unix ms, to: end unix ms };"
                " overrides `window` if both given."
            ),
        ] = None,
        apps: Annotated[
            list[str] | None,
            Field(description="Filter to specific source_apps, e.g. ['Slack','Microsoft Teams']."),
        ] = None,
        limit: Annotated[
            int, Field(ge=1, le=20, description="Max threads to return (default 10, max 20).")
        ] = 10,
    ) -> dict[str, Any]:
        return guarded(tools.active_threads, window, time_range, apps, limit)

    @server.tool(description=LATEST_DESC)
    def get_latest_context(
        hours: Annotated[
            float,
            Field(
                ge=0.25,
                le=168,
                description="Look back this many hours for recent activity (default 2, max 168).",
            ),
        ] = 2,
        apps: Annotated[
            list[str] | None,
            Field(description="Filter to specific source_apps, e.g. ['Slack','Apple Mail']."),
        ] = None,
        limit: Annotated[
            int, Field(ge=1, le=5, description="Max threads to return (default 3, max 5).")
        ] = 3,
    ) -> dict[str, Any]:
        return tools.latest_context(hours, apps, limit)

    @server.tool(description=MEETING_DESC)
    def meeting_memory(
        action: Annotated[
            Literal["list", "search", "get_context"],
            Field(
                description="'list' for counting/listing meetings, 'search' for semantic search"
                " within meeting content, 'get_context' for full transcript of a specific"
                " meeting."
            ),
        ],
        query: Annotated[
            str | None, Field(description="Semantic search query. Required for 'search'.")
        ] = None,
        thread_id: Annotated[
            str | None,
            Field(description="Meeting thread ID. Required for 'get_context'."),
        ] = None,
        time_range: Annotated[
            dict[str, Any] | None,
            Field(description="Time range filter: { last }, { day } or { day_range }."),
        ] = None,
        limit: Annotated[int | None, Field(ge=1, le=20)] = None,
        cursor: Annotated[int | None, Field(ge=0)] = None,
    ) -> dict[str, Any]:
        return tools.meetings(action, thread_id)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baseline behind minimi's connector tool surface")
    parser.add_argument("--db", required=True, help="database built by baseline.build")
    parser.add_argument("--now", required=True, help="ISO time with offset")
    parser.add_argument("--tz", default="Asia/Kolkata", help="IANA timezone reported as `tz`")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        parser.error(f"database not found: {args.db}")

    from baseline.embed import SentenceTransformerEmbedder

    tools = MinimiShapeTools(args.db, args.now, SentenceTransformerEmbedder(), args.tz)
    build_server(tools).run("stdio")


if __name__ == "__main__":
    main()
