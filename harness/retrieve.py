"""R1 to R10: answer a question with one packed context, no LLM at read time.

meta.now is the only clock. Current means valid_to is null and not expired at now.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

import numpy as np

from harness.store import OPEN_KINDS, Store
from harness.text import Embedder, dt, human_time, iso, overlap, same_person

RRF_K = 60
VECTOR_K = 40
BM25_K = 40
MMR_COSINE = 0.9
CHANGED_DAYS = 14
CHANGED_MAX = 5
RAW_MAX = 3
MAX_LINES = 25
MAX_RANKED_LINES = 8
SCORE_FLOOR = 0.35
RELATED_COSINE = 0.2
KEYWORD_K = 10
DEFAULT_TOKENS = 1500
APP_LABEL = {"screen": None, "gmail": "gmail", "calendar": "calendar"}
SCREEN_APP = {"whatsapp_web": "whatsapp", "gmail": "gmail"}

# R1: question words -> filter. Data, not code paths: extend by adding rows.
PARSE_ROWS: list[tuple[str, dict[str, Any]]] = [
    (r"\bowes? me\b|\bwaiting (on|for)\b", {"name": "todos others owe you", "kinds": ["todo"], "owner": "other"}),
    (r"\b(i|do i) (still )?owe\b|\bowe (people|anyone|anybody)\b|\bneed to\b|\bhave to\b|\bpromised\b|\bshould i do\b|\bto ?do\b",
     {"name": "your open todos", "kinds": ["todo"], "owner": "you"}),
    (r"\bagree[ds]?\b|\bcommit(ted|ments?)?\b", {"name": "what you agreed to", "kinds": ["todo", "event"]}),
    (r"\bpending\b|\boutstanding\b|\bopen\b", {"name": "open items", "kinds": list(OPEN_KINDS)}),
    (r"\bwhat'?s on\b|\bplans?\b|\bschedule\b|\bfree\b|\bbusy\b|\bdinner\b|\blunch\b|\bbrunch\b|\bmeeting\b",
     {"name": "events", "kinds": ["event"]}),
    (r"\blikes?\b|\bprefers?\b|\bfavou?rite\b|\ballergic\b|\beats?\b|\bvegetarian\b|\bfood\b|\bdiet\b|\brestrictions?\b",
     {"name": "preferences and facts", "kinds": ["preference", "fact"]}),
]


@dataclass
class Parsed:
    filters: list[dict[str, Any]] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    window: tuple[datetime, datetime] | None = None
    window_name: str | None = None


def time_window(q: str, now: datetime) -> tuple[tuple[datetime, datetime], str] | None:
    """Time words read against now. Explicit phrases win over a bare weekday."""
    tz = now.tzinfo
    today = now.date()

    def span(d0, d1):
        return (datetime.combine(d0, time.min, tz), datetime.combine(d1, time(23, 59, 59), tz))

    monday = today - timedelta(days=today.weekday())
    phrases = [
        (r"\btomorrow\b", lambda: span(today + timedelta(days=1), today + timedelta(days=1))),
        (r"\btoday\b|\btonight\b", lambda: span(today, today)),
        (r"\byesterday\b", lambda: span(today - timedelta(days=1), today - timedelta(days=1))),
        (r"\b(this )?weekend\b", lambda: span(*_weekend(today))),
        (r"\bnext week\b", lambda: span(monday + timedelta(days=7), monday + timedelta(days=13))),
        (r"\blast week\b", lambda: span(monday - timedelta(days=7), monday - timedelta(days=1))),
        (r"\bthis week\b", lambda: span(monday, monday + timedelta(days=6))),
        (r"\blast month\b", lambda: _last_month(today, tz)),
        (r"\bthis month\b", lambda: span(today.replace(day=1), today)),
    ]
    for pattern, fn in phrases:
        m = re.search(pattern, q)
        if m:
            return fn(), m.group(0)
    days = re.findall(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q)
    if len(set(days)) == 1:
        wd = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(days[0])
        d = today + timedelta(days=(wd - today.weekday()) % 7)
        return span(d, d), days[0]
    return None


def _weekend(today):
    if today.weekday() >= 5:
        sat = today - timedelta(days=today.weekday() - 5)
    else:
        sat = today + timedelta(days=5 - today.weekday())
    return sat, sat + timedelta(days=1)


def _last_month(today, tz):
    first = today.replace(day=1)
    last_prev = first - timedelta(days=1)
    return (datetime.combine(last_prev.replace(day=1), time.min, tz),
            datetime.combine(last_prev, time(23, 59, 59), tz))


class Retriever:
    def __init__(self, store: Store, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    @property
    def now(self) -> str:
        return self.store.get_meta("now")

    def known_people(self) -> list[str]:
        names: set[str] = set()
        for r in self.store.db.execute("SELECT DISTINCT people FROM memories"):
            for p in json.loads(r[0]):
                names.add(p)
        return sorted(names)

    # R1

    def parse(self, question: str, people: list[str] | None = None,
              time_range: dict[str, str] | None = None) -> Parsed:
        q = question.lower().replace("’", "'")
        out = Parsed()
        for pattern, f in PARSE_ROWS:
            if re.search(pattern, q):
                out.filters.append(dict(f))
        now = dt(self.now)
        if time_range and (time_range.get("start") or time_range.get("end")):
            start = dt(time_range["start"]) if time_range.get("start") else now - timedelta(days=3650)
            end = dt(time_range["end"]) if time_range.get("end") else now + timedelta(days=3650)
            out.window, out.window_name = (start, end), "given range"
        else:
            tw = time_window(q, now)
            if tw:
                out.window, out.window_name = tw
        words = re.findall(r"[a-z][a-z.']+", q)
        found = []
        for name in self.known_people():
            first = name.lower().split()[0].rstrip("'s")
            if any(w.rstrip("'s") == first or w.split("'")[0] == first for w in words):
                found.append(name)
        if people:
            found += [p for k in people for p in self.known_people() if same_person(p, k)] or people
        out.people = sorted(set(found))
        return out

    # helpers

    def _in_window(self, m: dict[str, Any], w) -> bool:
        if not w:
            return True
        for key in ("event_at", "valid_from"):
            if m.get(key) and w[0] <= dt(m[key]) <= w[1]:
                return True
        return False

    def _match(self, m, f: dict[str, Any], parsed: Parsed) -> bool:
        if f.get("kinds") and m["kind"] not in f["kinds"]:
            return False
        if f.get("owner") and m["owner"] != f["owner"]:
            return False
        if parsed.people and not overlap(m["people"], parsed.people):
            return False
        return self._in_window(m, parsed.window)

    def _past_window(self, parsed: Parsed) -> bool:
        return bool(parsed.window and parsed.window[1] <= dt(self.now))

    def time_fit(self, m: dict[str, Any], parsed: Parsed) -> float:
        now = dt(self.now)
        if parsed.window:
            return 1.0 if self._in_window(m, parsed.window) else 0.3
        ev = dt(m["event_at"]) if m["event_at"] else None
        if m["kind"] == "event" and ev and ev < now:
            return 0.2
        if m["kind"] in OPEN_KINDS and ev:
            days = (ev - now).total_seconds() / 86400
            if days <= 2:
                return 1.0
            return max(0.5, 1.0 - 0.5 * (days - 2) / 12)
        if m["kind"] in ("fact", "preference"):
            return 1.0
        age = (now - dt(m["valid_from"])).total_seconds() / 86400
        if m["kind"] == "episode":
            return 0.5 ** (age / 30)
        if m["kind"] == "todo":
            return 0.5 ** (age / 14)
        return 1.0

    # R2 to R5 over one pool of memories

    def rank(self, question: str, qvec, pool: list[dict[str, Any]], parsed: Parsed,
             filter_lists: list[list[int]]) -> list[tuple[float, dict[str, Any]]]:
        if not pool:
            return []
        ids = [m["id"] for m in pool]
        by_id = {m["id"]: m for m in pool}
        lists = [[i for i, _ in self.store.vector_top(qvec, ids, VECTOR_K)],
                 self.store.bm25_memories(question, BM25_K, set(ids))]
        lists += [[i for i in fl if i in by_id] for fl in filter_lists]
        rrf: dict[int, float] = {}
        for lst in lists:
            for rank, i in enumerate(lst):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        if not rrf:
            return []
        top = max(rrf.values())
        scored = sorted(((s / top * self.time_fit(by_id[i], parsed), by_id[i]) for i, s in rrf.items()),
                        key=lambda x: -x[0])
        picked: list[tuple[float, dict[str, Any]]] = []
        vecs: list[np.ndarray] = []
        for s, m in scored:
            v = self.store.vector(m["id"])
            if any(float(v @ p) > MMR_COSINE for p in vecs):
                continue
            picked.append((s, m))
            vecs.append(v)
        return picked

    @staticmethod
    def exhaustive(f: dict[str, Any], parsed: Parsed) -> bool:
        """An owner or a time window makes a filter an enumeration ("what do I owe", "this
        weekend"): every hit is shown and counted. Kind-only or people-only filters only rank."""
        return bool(f.get("owner") or parsed.window)

    def filter_list(self, f, parsed: Parsed, pool, qvec=None) -> list[int]:
        hits = [m for m in pool if self._match(m, f, parsed)]
        if qvec is not None and not self.exhaustive(f, parsed):
            hits.sort(key=lambda m: -float(self.store.vector(m["id"]) @ qvec))
        else:
            hits.sort(key=lambda m: (m["event_at"] is None, m["event_at"] or "", m["id"]))
        return [m["id"] for m in hits]

    @staticmethod
    def _without_names(question: str, people: list[str]) -> str:
        out = question
        for p in people:
            for tok in p.split():
                out = re.sub(r"\b" + re.escape(tok) + r"('s)?\b", "", out, flags=re.I)
        return " ".join(out.split())

    # R6 and R9 formatting

    def sources_label(self, mid: int) -> tuple[str, str | None]:
        srcs = self.store.memory_sources(mid)
        apps = []
        for s in srcs:
            a = SCREEN_APP.get(s["app"], s["app"]) if s["kind"] == "screen" else s["kind"]
            if a not in apps:
                apps.append(a)
        quote = next((s["quote"] for s in srcs if s["role"] == "created" and s["quote"]), None)
        return "+".join(apps), quote

    def history(self, m: dict[str, Any]) -> str:
        for link in self.store.links_from(m["id"]):
            if link["type"] != "replaces":
                continue
            old = self.store.memory(link["to_id"])
            if not old or old["valid_from"] >= m["valid_from"]:
                continue
            timed = old["event_at"] and m["event_at"] and old["event_at"] != m["event_at"]
            what = human_time(old["event_at"]) if timed else old["text"]
            by = self.store.source(link["source_id"]) if link["source_id"] else None
            how = f" by {_whose(by['author'])} {self._medium(by)}" if by else ""
            return f" (was: {what}, changed {human_time(link['at'], False)}{how})"
        return ""

    @staticmethod
    def _medium(s: dict[str, Any]) -> str:
        if s["kind"] == "calendar":
            return "calendar invite"
        if s["app"] in ("gmail", "gmail_connector"):
            return "email"
        return "WhatsApp message" if s["app"] == "whatsapp_web" else "message"

    def line(self, m: dict[str, Any], conflicts: dict[int, list[int]]) -> str:
        now = self.now
        kind = m["kind"] + (f" ({m['owner']})" if m["owner"] else "")
        parts = [f"[m{m['id']}] {kind}"]
        if m["event_at"]:
            due = human_time(m["event_at"])
            if m["kind"] == "todo" and m["valid_to"] is None and m["event_at"] < now:
                due = f"OVERDUE since {due}"
            elif m["kind"] == "todo":
                due = f"due {due}"
            parts.append(due)
        text = m["text"] + self.history(m)
        if m["valid_to"] is not None:
            text += f" [{m['end_reason']} {human_time(m['valid_to'], False)}]"
        parts.append(text)
        apps, quote = self.sources_label(m["id"])
        if quote:
            parts.append(f'"{quote}"')
        if apps:
            parts.append(apps)
        if m["id"] in conflicts:
            parts.append("possible conflict with " + ", ".join(f"[m{i}]" for i in conflicts[m["id"]]))
        return " · ".join(parts)

    def conflicts(self, current: list[dict[str, Any]]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        events = [m for m in current if m["kind"] == "event" and m["event_at"]]
        for i, a in enumerate(events):
            for b in events[i + 1 :]:
                if a["event_at"][:10] == b["event_at"][:10] and overlap(a["people"], b["people"]):
                    out.setdefault(a["id"], []).append(b["id"])
                    out.setdefault(b["id"], []).append(a["id"])
        ids = {m["id"] for m in current}
        for r in self.store.db.execute("SELECT target_id, memory_id FROM ops_log WHERE op = 'downgraded_pair'"):
            a, b = r["target_id"], r["memory_id"]
            if a in ids and b in ids:
                out.setdefault(a, []).append(b)
                out.setdefault(b, []).append(a)
        return out

    def changed_line(self, m: dict[str, Any]) -> str:
        reason = m["end_reason"]
        srcs = self.store.memory_sources(m["id"])
        closer = next((s for s in reversed(srcs) if s["role"] == "ended"), None)
        replacement = next((self.store.memory(lk["from_id"]) for lk in self.store.links_to(m["id"])
                            if lk["type"] == "replaces"), None)
        when = human_time(m["valid_to"], False)
        bits = [f"[m{m['id']}] {m['kind']}"]
        if m["event_at"]:
            bits.append(human_time(m["event_at"]))
        bits.append(m["text"])
        if replacement:
            by = self.store.memory_sources(replacement["id"])
            src = next((s for s in by if s["role"] == "created"), None)
            how = f" by {_whose(src['author'])} {self._medium(src)}: \"{src['quote'] or src['text']}\"" if src else ""
            bits.append(f"replaced {when} by [m{replacement['id']}]{how}")
        elif closer:
            bits.append(f"{reason} {when} by {_whose(closer['author'])} {self._medium(closer)}: \"{closer['quote'] or closer['text']}\"")
        else:
            bits.append(f"{reason} {when}")
        return " · ".join(bits)

    # R1 to R10

    def get_context(self, question: str, time_range: dict[str, str] | None = None,
                    people: list[str] | None = None, max_tokens: int = DEFAULT_TOKENS) -> str:
        self.store.refresh()
        now = self.now
        parsed = self.parse(question, people, time_range)
        qvec = self.embedder.one(question)
        current = self.store.current(now)
        pool = list(current)
        if self._past_window(parsed):
            pool += [m for m in self.store.memories("valid_to IS NOT NULL AND end_reason IN ('ended', 'expired')")
                     if self._in_window(m, parsed.window)]
        filters = list(parsed.filters)
        if parsed.people and not filters:
            filters.append({"name": "about " + ", ".join(parsed.people)})
        if parsed.window and not filters:
            filters.append({"name": f"items {parsed.window_name}"})
        filter_lists = [self.filter_list(f, parsed, pool, qvec) for f in filters]
        ranked = self.rank(question, qvec, pool, parsed, filter_lists)
        must = {i for f, fl in zip(filters, filter_lists, strict=True) if self.exhaustive(f, parsed) for i in fl}
        content = self._without_names(question, parsed.people)
        cvec = self.embedder.one(content) if content != question else qvec
        keyword = set(self.store.bm25_memories(content, KEYWORD_K, {m["id"] for m in pool}))
        kept, extra = [], 0
        for s, m in ranked:
            if m["id"] in must:
                kept.append((s, m))
            elif extra < MAX_RANKED_LINES and s >= SCORE_FLOOR and (
                    m["id"] in keyword or float(self.store.vector(m["id"]) @ cvec) >= RELATED_COSINE):
                kept.append((s, m))
                extra += 1
        ranked = kept

        conflicts = self.conflicts(current)
        budget = max_tokens * 4
        header = f"as_of: {_as_of(now)}"
        lines: list[str] = []
        shown: set[int] = set()
        used = len(header)
        for _s, m in ranked:
            if len(lines) >= MAX_LINES:
                break
            ln = self.line(m, conflicts)
            if used + len(ln) > budget * 0.8:
                break
            lines.append(ln)
            shown.add(m["id"])
            used += len(ln)

        # R7
        cutoff = iso(dt(now) - timedelta(days=CHANGED_DAYS))
        changed_pool = self.store.memories(
            "valid_to IS NOT NULL AND valid_to >= ? AND valid_to <= ? AND end_reason IN ('replaced', 'ended', 'expired')",
            (cutoff, now),
        )
        changed_pool = [m for m in changed_pool if m["id"] not in shown
                        and (not parsed.window or self._in_window(m, parsed.window))]
        changed_ranked = self.rank(question, qvec, changed_pool, parsed,
                                   [self.filter_list(f, parsed, changed_pool) for f in filters])
        changed: list[str] = []
        for _s, m in changed_ranked:
            if len(changed) >= CHANGED_MAX:
                break
            relevant = (float(self.store.vector(m["id"]) @ qvec) > 0.35
                        or (parsed.people and overlap(m["people"], parsed.people))
                        or any(m["id"] in fl for fl in [self.filter_list(f, parsed, [m]) for f in parsed.filters]))
            if not relevant:
                continue
            repl = [lk["from_id"] for lk in self.store.links_to(m["id"]) if lk["type"] == "replaces"]
            if repl and all(r in shown for r in repl) and not parsed.filters:
                continue
            ln = self.changed_line(m)
            if used + len(ln) > budget * 0.95:
                break
            changed.append(ln)
            used += len(ln)

        # R8
        raw = self.store.bm25_sources(
            question, RAW_MAX,
            "s.skip_reason IS NULL AND s.burst_id IN (SELECT id FROM bursts WHERE status IN ('open', 'ready'))",
        )

        # R10
        footer = []
        for f, fl in zip(filters, filter_lists, strict=True):
            if not self.exhaustive(f, parsed):
                continue
            n_shown = sum(1 for i in fl if i in shown)
            footer.append(f"{f['name']}: {n_shown} of {len(fl)} shown")
            if n_shown < len(fl):
                args = []
                if f.get("kinds") and len(f["kinds"]) == 1:
                    args.append(f'kind="{f["kinds"][0]}"')
                if f.get("owner"):
                    args.append(f'owner="{f["owner"]}"')
                if parsed.people:
                    args.append(f"people={parsed.people!r}")
                if parsed.window:
                    args.append(f'after="{iso(parsed.window[0])}", before="{iso(parsed.window[1])}"')
                footer.append(f'  more: minimi_search_memory(query="", {", ".join(args)})')

        out = [header, ""]
        out += lines or ["No current memories match."]
        if changed:
            out += ["", "Changed recently:"] + changed
        if raw:
            out += ["", "Not processed yet (raw messages):"]
            out += [f"[raw s{s['id']}] {human_time(s['said_at'])} · {s['author']} in {s['thread_title']}: \"{s['text']}\"" for s in raw]
        if footer:
            out += ["", "Footer:"] + footer
        return "\n".join(out)

    # minimi_search_memory

    def search(self, query: str = "", kind: str | None = None, owner: str | None = None,
               people: list[str] | None = None, after: str | None = None, before: str | None = None,
               include_history: bool = False, limit: int = 25, cursor: int = 0) -> dict[str, Any]:
        self.store.refresh()
        now = self.now
        pool = self.store.memories() if include_history else self.store.current(now)
        if kind:
            pool = [m for m in pool if m["kind"] == kind]
        if owner:
            pool = [m for m in pool if m["owner"] == owner]
        if people:
            pool = [m for m in pool if overlap(m["people"], people)]
        if after or before:
            w = (dt(after) if after else datetime.min.replace(tzinfo=dt(now).tzinfo),
                 dt(before) if before else datetime.max.replace(tzinfo=dt(now).tzinfo))
            pool = [m for m in pool if self._in_window(m, w)]
        if query.strip():
            parsed = Parsed()
            ranked = [m for _, m in self.rank(query, self.embedder.one(query), pool, parsed, [])]
        else:
            ranked = sorted(pool, key=lambda m: (m["event_at"] is None, m["event_at"] or m["valid_from"]))
        page = ranked[cursor : cursor + limit]
        conflicts = self.conflicts(self.store.current(now))
        nxt = cursor + limit if cursor + limit < len(ranked) else None
        return {
            "as_of": _as_of(now),
            "total_count": len(ranked),
            "has_more": nxt is not None,
            "next_cursor": nxt,
            "results": "\n".join(self.line(m, conflicts) for m in page) or "No matches.",
        }

    # minimi_get_source

    def get_source(self, memory_ids: list[str]) -> str:
        self.store.refresh()
        out = []
        for raw_id in memory_ids[:10]:
            mid = int(str(raw_id).lstrip("m[").rstrip("]"))
            m = self.store.memory(mid)
            if not m:
                out.append(f"[m{mid}] not found")
                continue
            out.append(self.line(m, {}))
            for s in self.store.memory_sources(mid):
                out.append(f"  {s['role']}: \"{s['quote'] or ''}\" in {s['thread_title']} ({self._medium(s)})")
                around = self.store.db.execute(
                    "SELECT * FROM (SELECT id, author, said_at, text FROM sources WHERE thread_key = ?"
                    " AND skip_reason IS NULL AND said_at < ? ORDER BY said_at DESC LIMIT 2)"
                    " UNION ALL SELECT id, author, said_at, text FROM sources WHERE id = ?"
                    " UNION ALL SELECT * FROM (SELECT id, author, said_at, text FROM sources WHERE thread_key = ?"
                    " AND skip_reason IS NULL AND said_at > ? ORDER BY said_at LIMIT 2)",
                    (s["thread_key"], s["said_at"], s["id"], s["thread_key"], s["said_at"]),
                ).fetchall()
                for r in sorted(around, key=lambda r: r["said_at"]):
                    mark = ">>" if r["id"] == s["id"] else "  "
                    out.append(f"    {mark} {human_time(r['said_at'])} {r['author']}: {r['text']}")
            out.append("")
        return "\n".join(out).strip()


def _whose(author: str) -> str:
    return "your" if author.strip().lower() in ("you", "me") else f"{author}'s"


def _as_of(now: str) -> str:
    d = dt(now)
    h = d.hour % 12 or 12
    return f"{d:%a} {d:%b} {d.day} {d.year}, {h}:{d.minute:02d} {'AM' if d.hour < 12 else 'PM'} IST (Asia/Kolkata)"
