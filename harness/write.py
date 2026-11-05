"""W4 to W8: calendar shortcut, candidates, one reconcile call per burst, validation, apply.

The write path never reads the clock. Every time check uses said_at of the cited message.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from harness.llm import LLM, LLMError
from harness.store import OPEN_KINDS, KINDS, Store
from harness.text import (
    Embedder, dt, fmt_day, human_time, is_you, iso, overlap, parse_event_at, quote_ok,
    rewrite_dates, snap_weekday, weekday_in,
)

MAX_CANDIDATES = 40
THREAD_CAP = 10
PEOPLE_CAP = 15
PER_MESSAGE = 3
CONTEXT_MESSAGES = 3
THREAD_ENDED_DAYS = 14
STATUS_DAYS = 7
APP_NAMES = {"whatsapp_web": "WhatsApp", "gmail": "Gmail", "gmail_connector": "Gmail", "calendar": "Calendar"}

PROMPT = """You keep a personal memory for one user, called "You". You get a chunk of new messages from one conversation, a few earlier messages for context, and existing memories that might be related. Decide what to write. Reply with JSON only.

A memory is one self-contained sentence from the user's point of view with absolute dates, like "You'll send Priya the deck by Fri Nov 13, 6:00 PM" or "Dinner with Sam on Fri Nov 20 at 9:30 PM".

Kinds: event (something planned at a time), todo (something someone agreed to do), fact, preference, status (a temporary state expected to pass within days, like being sick, traveling this week or mail being down), episode (something that happened). Lasting changes, like a new diet, job or address, are facts or preferences.
owner, for todos only: "you" if the user has to act, "other" if someone else has to.

Ops:
- new: nothing in the candidates covers it. If a candidate shows it is already done or no longer true, add closed_by with that candidate's id.
- same: says what candidate cX already says. A copy of an ended candidate stays on that ended candidate.
- replaces: candidate cX is no longer true; give the new version.
- extends: adds detail to cX; both stay true.
- ends: cX is done, cancelled, or stopped being true, with no replacement.

Rules:
- Every op cites one message as msg (b1, b2, ...) and a quote copied word for word from that message.
- "You" in a memory always means the user. When someone else writes "my" or "I", it is about them: write their name ("Priya's flight", not "your flight").
- Resolve relative dates ("Friday", "tomorrow", "next Friday") against the cited message's own date, never today. Use the calendar given with the messages to look up weekdays; don't compute them yourself.
- A time without AM or PM takes the obvious meaning from context: dinner, drinks and "tonight" are evening, breakfast and brunch are morning.
- An event that simply happened is not ended. Only end an event that was cancelled or replaced.
- Keep exact identifiers (flight numbers, amounts, file names) in the memory text.
- A request becomes a todo for you only when you agree to it. A proposal ("can we do 9:30?") is not an agreement until someone accepts it.
- A todo is a commitment between people: someone asked and the other agreed, or someone promised another person something. Talking about what you're doing or about to do ("time to refill the printer", "nap time") is not a todo.
- "Same as last time" and similar phrases: work out the real value from the candidates. "Last time" is the most recent earlier occurrence by its date (event_at), in its current version, not the usual or most frequent one.
- The same thing often arrives again from another app (an email copy, a calendar entry). If a candidate already says it, use same, even when the candidate is ended.
- Most chat is small talk: skip it unless it closes or changes something. Opinions about movies, memes, sports or the news, jokes, reactions ("haha", "riveting drama"), and passing remarks are small talk. Write a fact or preference only if it would still matter to the user weeks later, like a diet, an allergy, a job, an address, a relationship, or an identifier.
- Pure chat needs no ops; an empty ops list is fine for it. But when a message agrees to, repeats or confirms something a candidate already says (from any app), always give a same op for it, so the memory keeps every source.
- Give the end state of the chunk: if something changed twice inside the chunk, report only the final version.
- event_at is "YYYY-MM-DDTHH:MM" for a time, or "YYYY-MM-DD" for a due date without a time.

JSON shape:
{"ops": [{"op": "new|same|replaces|extends|ends", "target": "c3 or null", "msg": "b1", "quote": "...", "text": "...", "kind": "...", "owner": "you|other|null", "people": ["..."], "event_at": "YYYY-MM-DDTHH:MM or null", "closed_by": "c5 or null"}]}
For same and ends, text, kind, people and event_at may be null.
"""


class Writer:
    def __init__(self, store: Store, llm: LLM, embedder: Embedder) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder

    # W5

    def burst_people(self, thread_key: str, msgs: list[dict[str, Any]]) -> list[str]:
        people = {m["author"] for m in msgs if not is_you(m["author"])}
        for m in msgs:
            people.update(m["participants"])
        authors = {
            r[0] for r in self.store.db.execute(
                "SELECT DISTINCT author FROM sources WHERE thread_key = ? AND skip_reason IS NULL",
                (thread_key,),
            )
            if not is_you(r[0])
        }
        if len(authors) == 1:
            people |= authors
        return sorted(people)

    def candidates(self, thread_key: str, msgs: list[dict[str, Any]], people: list[str],
                   vecs) -> list[dict[str, Any]]:
        start = msgs[0]["said_at"]
        picked: list[dict[str, Any]] = []
        seen: set[int] = set()

        def add(mems, cap):
            n = 0
            for m in mems:
                if len(picked) >= MAX_CANDIDATES or n >= cap:
                    return
                if m["id"] not in seen:
                    seen.add(m["id"])
                    picked.append(m)
                    n += 1

        cutoff = iso(dt(start) - timedelta(days=THREAD_ENDED_DAYS))
        thread = [self.store.memory(i) for i in self.store.thread_memory_ids(thread_key)]
        thread = [m for m in thread if m["valid_to"] is None or m["valid_to"] >= cutoff]
        add(sorted(thread, key=lambda m: m["valid_from"], reverse=True), THREAD_CAP)

        everything = self.store.memories()
        if people:
            mine = [m for m in everything if overlap(m["people"], people)]
            later = [m for m in mine if m["valid_from"] > start]
            open_ = [m for m in mine if m["valid_to"] is None and m["kind"] in OPEN_KINDS]
            later.sort(key=lambda m: m["valid_from"])
            open_ = [m for m in open_ if m["kind"] == "todo"] + sorted(
                (m for m in open_ if m["kind"] != "todo"), key=lambda m: m["valid_from"], reverse=True
            )
            add(later + open_, PEOPLE_CAP)

        current_ids = [m["id"] for m in everything if m["valid_to"] is None]
        allowed = set(current_ids)
        by_id = {m["id"]: m for m in everything}
        if people:
            about = [m["id"] for m in everything if m["valid_to"] is None and overlap(m["people"], people)]
            for v in vecs:
                add([by_id[i] for i, _ in self.store.vector_top(v, about, PER_MESSAGE)], PER_MESSAGE)
        for m, v in zip(msgs, vecs, strict=True):
            add([by_id[i] for i, _ in self.store.vector_top(v, current_ids, PER_MESSAGE)], PER_MESSAGE)
            add([by_id[i] for i in self.store.bm25_memories(m["text"], PER_MESSAGE, allowed)], PER_MESSAGE)
        return picked

    # W6

    def build_prompt(self, thread_key: str, msgs, cands, note: str | None = None) -> str:
        first = msgs[0]
        title = first["thread_title"]
        app = APP_NAMES.get(first["app"], first["app"])
        ctx = self.store.db.execute(
            "SELECT author, said_at, text FROM sources WHERE thread_key = ? AND skip_reason IS NULL"
            " AND said_at < ? ORDER BY said_at DESC LIMIT ?",
            (thread_key, first["said_at"], CONTEXT_MESSAGES),
        ).fetchall()[::-1]
        lines = [f"Conversation: {title} ({app})", "", "Earlier messages (context only):"]
        lines += [f"- {fmt_day(r['said_at'])} {r['author']}: {r['text']}" for r in ctx] or ["- none"]
        d0 = dt(first["said_at"]).date()
        cal = [d0 + timedelta(days=i) for i in range(-7, 15)]
        lines += ["", "Calendar: " + ", ".join(d.strftime("%a %Y-%m-%d") for d in cal)]
        lines += ["", "New messages:"]
        lines += [f"b{i + 1} · {fmt_day(m['said_at'])} · {m['author']}: {m['text']}" for i, m in enumerate(msgs)]
        lines += ["", "Existing memories:"]
        for i, m in enumerate(cands):
            status = "current" if m["valid_to"] is None else f"ENDED ({m['end_reason']})"
            due = f" · at {m['event_at'][:16]}" if m["event_at"] else ""
            own = f" ({m['owner']})" if m["owner"] else ""
            ppl = ", ".join(m["people"]) or "-"
            lines.append(f"c{i + 1} [{status}] {m['kind']}{own}{due} · people: {ppl} · said {m['valid_from'][:16]} · {m['text']}")
        if not cands:
            lines.append("none")
        if note:
            lines += ["", note]
        return "\n".join(lines)

    # W4 to W8 for one burst

    def process(self, burst_id: int, note: str | None = None) -> dict[str, Any]:
        b = self.store.db.execute("SELECT * FROM bursts WHERE id = ?", (burst_id,)).fetchone()
        msgs = self.store.sources("burst_id = ?", (burst_id,))
        report: dict[str, Any] = {"burst": burst_id, "thread": b["thread_key"],
                                  "labels": [m["label"] for m in msgs], "ops": []}
        self.store.db.execute("UPDATE bursts SET attempts = attempts + 1 WHERE id = ?", (burst_id,))
        if not msgs:
            self._finish(burst_id, "done")
            return report
        thread_key = b["thread_key"]
        people = self.burst_people(thread_key, msgs)
        vecs = self.embedder.many([m["text"] for m in msgs])
        cands = self.candidates(thread_key, msgs, people, vecs)

        if all(m["kind"] == "calendar" for m in msgs):
            handled = self.calendar_shortcut(burst_id, thread_key, msgs, cands, report)
            if handled:
                self._finish(burst_id, "done")
                return report

        prompt = self.build_prompt(thread_key, msgs, cands, note)
        try:
            out = self.llm.json(PROMPT, prompt)
        except LLMError as exc:
            self.store.log(msgs[-1]["said_at"], "write", "failed", burst_id=burst_id, reason=str(exc))
            self._finish(burst_id, "failed", str(exc))
            report["error"] = str(exc)
            return report
        except (BotoCoreError, ClientError) as exc:  # network or throttling after retries; the dream retries it
            self.store.log(msgs[-1]["said_at"], "write", "failed", burst_id=burst_id, reason=repr(exc))
            self._finish(burst_id, "failed", repr(exc))
            report["error"] = repr(exc)
            return report

        ops = [o for o in out.get("ops", []) if isinstance(o, dict)]
        bmap = {f"b{i + 1}": m for i, m in enumerate(msgs)}
        cmap = {f"c{i + 1}": m["id"] for i, m in enumerate(cands)}
        ops.sort(key=lambda o: bmap.get(str(o.get("msg")), msgs[-1])["said_at"])
        rejected = 0
        for op in ops:
            res = self.apply(op, bmap, cmap, burst_id, made_by="write")
            report["ops"].append(res)
            rejected += res["outcome"] == "rejected"
        self._finish(burst_id, "retry" if rejected else "done")
        return report

    def _finish(self, burst_id: int, status: str, note: str | None = None) -> None:
        self.store.db.execute(
            "UPDATE bursts SET status = ?, note = ? WHERE id = ?", (status, note, burst_id)
        )
        self.store.commit()

    def calendar_shortcut(self, burst_id, thread_key, msgs, cands, report) -> bool:
        """W4. A cancellation ends the event's memories. A version with no related memory becomes
        a templated memory with no LLM call. Anything else goes to the reconcile call."""
        last = msgs[-1]
        event = last["extra"].get("event") or {}
        if event.get("status") == "cancelled":
            for mid in self.store.thread_memory_ids(thread_key):
                m = self.store.memory(mid)
                if m["valid_to"] is None and m["kind"] == "event":
                    self.store.end_memory(mid, last["said_at"], "ended")
                    self.store.attach_source(mid, last["id"], "cancelled", "ended")
                    self.store.log(last["said_at"], "calendar", "applied", burst_id=burst_id,
                                   op="ends", target_id=mid)
                    report["ops"].append({"op": "ends", "outcome": "applied", "target": mid})
            return True
        if cands or not event.get("start"):
            return False
        people = [p for p in event.get("attendees", []) if not is_you(p)]
        start = parse_event_at(event["start"], last["said_at"])
        text = f"{event.get('title') or 'Calendar event'}, {human_time(start)}"
        if event.get("location"):
            text += f", at {event['location']}"
        mid = self._insert({"text": text, "kind": "event", "owner": None, "people": people,
                            "event_at": start, "event_end": parse_event_at(event.get("end"), last["said_at"]),
                            "valid_from": last["said_at"], "created_burst": burst_id},
                           last["id"], event.get("title"))
        self.store.log(last["said_at"], "calendar", "applied", burst_id=burst_id, op="new", memory_id=mid)
        report["ops"].append({"op": "new", "outcome": "applied", "id": mid, "via": "calendar template"})
        return True

    # W7 and W8

    def _insert(self, m: dict[str, Any], sid: int, quote: str | None) -> int:
        mid = self.store.add_memory(m, self.embedder.one(m["text"]))
        self.store.attach_source(mid, sid, quote, "created")
        return mid

    def apply(self, op: dict[str, Any], bmap, cmap, burst_id: int, made_by: str) -> dict[str, Any]:
        kind = str(op.get("op") or "").lower()
        res: dict[str, Any] = {"op": kind, "msg": op.get("msg"), "target": op.get("target")}

        def reject(reason: str) -> dict[str, Any]:
            res.update(outcome="rejected", reason=reason)
            self.store.log(at, made_by, "rejected", burst_id=burst_id, op=kind, reason=reason, raw=op)
            return res

        msg = bmap.get(str(op.get("msg")))
        at = msg["said_at"] if msg else next(iter(bmap.values()))["said_at"]
        if kind not in ("new", "same", "replaces", "extends", "ends"):
            return reject("unknown op")
        if not msg:
            return reject("cited message not in burst")
        quote = op.get("quote") or ""
        if not quote_ok(quote, msg["text"]):
            return reject("quote not in message")
        target = None
        if kind != "new":
            tid = cmap.get(str(op.get("target")))
            if tid is None:
                return reject("target not a candidate")
            target = self.store.memory(tid)
        said = msg["said_at"]
        people = [p for p in (op.get("people") or []) if isinstance(p, str) and p and not is_you(p)]

        if kind in ("same", "replaces") and target["people"] and people and not overlap(target["people"], people):
            self.store.log(said, made_by, "downgraded", burst_id=burst_id, op=kind,
                           target_id=target["id"], reason="no shared person", raw=op)
            res["downgraded_from"] = kind
            kind = "new"
            res["op"] = "new"
            pair_target = target
            target = None
        else:
            pair_target = None

        if kind == "same":
            self.store.attach_source(target["id"], msg["id"], quote, "confirmed")
            self.store.log(said, made_by, "applied", burst_id=burst_id, op="same", target_id=target["id"])
            res.update(outcome="applied", id=target["id"])
            return res

        if kind == "ends":
            if said <= target["valid_from"]:
                return reject("ends is older than its target")
            if target["valid_to"] is not None:
                self.store.attach_source(target["id"], msg["id"], quote, "confirmed")
                res.update(outcome="applied", id=target["id"], note="already ended")
                return res
            self.store.end_memory(target["id"], said, "ended")
            self.store.attach_source(target["id"], msg["id"], quote, "ended")
            self.store.log(said, made_by, "applied", burst_id=burst_id, op="ends", target_id=target["id"])
            res.update(outcome="applied", id=target["id"])
            return res

        text = (op.get("text") or "").strip()
        if not text:
            return reject("no text")
        mkind = str(op.get("kind") or "fact").lower()
        if mkind not in KINDS:
            mkind = "fact"
        owner = None
        if mkind == "todo":
            owner = op.get("owner") if op.get("owner") in ("you", "other") else (
                "you" if is_you(msg["author"]) else "other")
        event_at = parse_event_at(op.get("event_at"), said)
        if event_at and mkind in ("event", "todo"):
            snapped = snap_weekday(event_at, quote, said) if weekday_in(quote) is not None else (
                snap_weekday(event_at, text, said))
            if snapped:
                text = rewrite_dates(text, event_at, snapped)
                res["weekday_snap"] = f"{event_at[:16]} -> {snapped[:16]}"
                event_at = snapped
        expires = iso(dt(said) + timedelta(days=STATUS_DAYS)) if mkind == "status" else None
        new = {"text": text, "kind": mkind, "owner": owner, "people": people, "event_at": event_at,
               "expires_at": expires, "valid_from": said, "created_burst": burst_id}

        closer = None
        if kind == "new" and op.get("closed_by"):
            cid = cmap.get(str(op.get("closed_by")))
            closer = self.store.memory(cid) if cid else None
            if closer and closer["valid_from"] > said:
                new["valid_to"], new["end_reason"] = closer["valid_from"], "ended"
                res["note"] = f"closed by m{closer['id']}"
            else:
                res["note"] = "closed_by ignored"
                closer = None
        if kind == "replaces" and said <= target["valid_from"]:
            new["valid_to"], new["end_reason"] = target["valid_from"], "replaced"
            res["note"] = "older than target, inserted ended"

        mid = self._insert(new, msg["id"], quote)
        res.update(outcome="applied", id=mid)
        if kind == "replaces":
            if new.get("valid_to"):
                self.store.link(target["id"], mid, "replaces", made_by, target["valid_from"], None)
            else:
                self.store.end_memory(target["id"], said, "replaced")
                self.store.link(mid, target["id"], "replaces", made_by, said, msg["id"])
        elif kind == "extends":
            self.store.link(mid, target["id"], "extends", made_by, said, msg["id"])
        if closer:
            self.store.link(closer["id"], mid, "ends", made_by, closer["valid_from"], None)
        if pair_target:
            self.store.log(said, made_by, "pair", burst_id=burst_id, op="downgraded_pair",
                           target_id=pair_target["id"], memory_id=mid)
        self.store.log(said, made_by, "applied", burst_id=burst_id, op=kind,
                       target_id=target["id"] if target else None, memory_id=mid, raw=op)
        return res
