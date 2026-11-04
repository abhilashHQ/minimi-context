"""D1 to D5: the background pass. Live it runs nightly or after 300 new memories; in replay it
runs once after ingest at the corpus's now. Every change keeps the time check: a memory can only
be ended or replaced by something said after it."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import numpy as np

from botocore.exceptions import BotoCoreError, ClientError

from harness.llm import LLM, LLMError
from harness.store import Store
from harness.text import Embedder, overlap
from harness.write import Writer

MERGE_COSINE = 0.92
BATCH = 10
CLOSE_BATCH = 1
EVIDENCE = 3

SETTLE_PROMPT = """You maintain a personal memory for one user, called "You". Some memories may describe the same thing or contradict each other. For each group, decide using the source messages and when they were said. Reply with JSON only.

Decisions for a pair (a, b):
- "a_replaces_b": b is no longer true because of what a says (a must be said later than b).
- "b_replaces_a": the reverse.
- "same": both say the same thing.
- "both_stand": they are different things, both true.

JSON shape: {"decisions": [{"group": "g1", "a": "m1", "b": "m2", "decision": "...", "why": "..."}]}
"""

CLOSE_PROMPT = """You maintain a personal memory for one user, called "You". Each open item below is a commitment, plan or state that is still marked open. Under it is later evidence from any conversation. Decide if the evidence shows the item is done, cancelled or no longer true. A reminder, a repeat of the request, or a mention is not closure. Someone thanking the user for doing exactly the thing, or the user sharing the result, is closure. People are often named differently in different places (a role in one message, a name in another); match on what the evidence says was done. Reply with JSON only.

JSON shape: {"closes": [{"item": "t1", "evidence": "e2", "quote": "words copied from that evidence", "why": "..."}]}
Only list items that are closed. An empty list is fine.
"""

MERGE_PROMPT = """You maintain a personal memory for one user, called "You". Each pair below is two memories with very similar wording. Decide for each: "same" (one thing said twice), "a_replaces_b" or "b_replaces_a" (a later one makes the other untrue), "extends" (one adds detail, both true), or "unrelated". Reply with JSON only.

JSON shape: {"pairs": [{"pair": "p1", "decision": "...", "why": "..."}]}
"""


class Dreamer:
    def __init__(self, store: Store, llm: LLM, embedder: Embedder, writer: Writer) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.writer = writer
        self.report: dict[str, Any] = {"expired": 0, "retried": 0, "retry_failed": 0, "settled": [],
                                       "closed": [], "merged": [], "calls": 0}

    def run(self, now: str) -> dict[str, Any]:
        start_calls = self.llm.stats.calls
        self.expire(now)
        self.retry()
        self.settle(now)
        self.close_stale(now)
        self.merge(now)
        self.store.set_meta("last_dream", now)
        self.store.commit()
        self.report["calls"] = self.llm.stats.calls - start_calls
        return self.report

    # helpers

    def _quote_line(self, m: dict[str, Any]) -> str:
        srcs = self.store.memory_sources(m["id"])
        first = next((s for s in srcs if s["role"] == "created"), srcs[0] if srcs else None)
        if not first:
            return ""
        return f' (from {first["author"]} in {first["thread_title"]}, {first["said_at"][:16]}: "{first["text"]}")'

    def _describe(self, label: str, m: dict[str, Any]) -> str:
        when = f" · at {m['event_at'][:16]}" if m["event_at"] else ""
        own = f" ({m['owner']})" if m["owner"] else ""
        return f"{label} {m['kind']}{own}{when} · said {m['valid_from'][:16]} · {m['text']}{self._quote_line(m)}"

    def _replace(self, newer: dict[str, Any], older: dict[str, Any], now: str, why: str) -> bool:
        if newer["valid_from"] <= older["valid_from"] or older["valid_to"] is not None:
            return False
        self.store.end_memory(older["id"], newer["valid_from"], "replaced")
        self.store.link(newer["id"], older["id"], "replaces", "dream", newer["valid_from"], None)
        self.store.log(now, "dream", "applied", op="replaces", target_id=older["id"],
                       memory_id=newer["id"], reason=why)
        return True

    def _same(self, keep: dict[str, Any], drop: dict[str, Any], now: str, why: str) -> None:
        self.store.end_memory(drop["id"], now, "merged")
        self.store.link(drop["id"], keep["id"], "merged_into", "dream", now, None)
        for s in self.store.memory_sources(drop["id"]):
            self.store.attach_source(keep["id"], s["id"], s["quote"], "confirmed")
        self.store.log(now, "dream", "applied", op="merge", target_id=drop["id"], memory_id=keep["id"], reason=why)

    def _pair(self, a: dict[str, Any], b: dict[str, Any], decision: str, now: str, why: str) -> str:
        a, b = self.store.memory(a["id"]), self.store.memory(b["id"])
        if a["valid_to"] is not None or b["valid_to"] is not None:
            return "skipped, already ended"
        if decision == "a_replaces_b":
            return "applied" if self._replace(a, b, now, why) else "rejected by time check"
        if decision == "b_replaces_a":
            return "applied" if self._replace(b, a, now, why) else "rejected by time check"
        if decision == "same":
            keep, drop = sorted((a, b), key=lambda m: m["valid_from"])
            self._same(keep, drop, now, why)
            return "applied"
        if decision == "extends":
            newer, older = sorted((a, b), key=lambda m: m["valid_from"], reverse=True)
            self.store.link(newer["id"], older["id"], "extends", "dream", now, None)
            return "applied"
        return "no change"

    def _call(self, system: str, user: str) -> dict[str, Any] | None:
        try:
            return self.llm.json(system, user, effort="high")
        except (LLMError, BotoCoreError, ClientError) as exc:  # a failed dream step leaves the store as it was
            self.store.log("dream", "dream", "failed", reason=repr(exc))
            return None

    # D1

    def expire(self, now: str) -> None:
        for m in self.store.memories("valid_to IS NULL AND expires_at IS NOT NULL AND expires_at <= ?", (now,)):
            self.store.end_memory(m["id"], m["expires_at"], "expired")
            self.store.log(now, "dream", "applied", op="expire", target_id=m["id"])
            self.report["expired"] += 1

    # D2

    def retry(self) -> None:
        rows = self.store.db.execute(
            "SELECT id, status FROM bursts WHERE status IN ('failed', 'retry') AND attempts < 2"
        ).fetchall()
        for r in rows:
            rejected = self.store.db.execute(
                "SELECT op, reason, raw_json FROM ops_log WHERE burst_id = ? AND outcome = 'rejected'", (r["id"],)
            ).fetchall()
            note = (
                "Note: an earlier pass over these messages already applied some ops; their memories are in the"
                " candidates, so use same for those. These ops were rejected: "
                + "; ".join(f"{x['op']} ({x['reason']})" for x in rejected)
                if rejected else "Note: an earlier pass over these messages failed. Try again."
            )
            rep = self.writer.process(r["id"], note)
            self.report["retried"] += 1
            if rep.get("error") or any(o["outcome"] == "rejected" for o in rep["ops"]):
                self.report["retry_failed"] += 1

    # D3

    def settle(self, now: str) -> None:
        pairs: list[tuple[dict, dict]] = []
        for r in self.store.db.execute(
            "SELECT target_id, memory_id FROM ops_log WHERE op = 'downgraded_pair'"
        ).fetchall():
            a, b = self.store.memory(r["memory_id"]), self.store.memory(r["target_id"])
            if a and b and a["valid_to"] is None and b["valid_to"] is None:
                pairs.append((a, b))
        events = [m for m in self.store.current(now) if m["kind"] == "event" and m["event_at"]]
        for a, b in combinations(events, 2):
            if a["event_at"][:10] == b["event_at"][:10] and overlap(a["people"], b["people"]):
                pairs.append((a, b))
        seen: set[tuple[int, int]] = set()
        uniq = []
        for a, b in pairs:
            key = tuple(sorted((a["id"], b["id"])))
            if key not in seen:
                seen.add(key)
                uniq.append((a, b))
        for chunk in _chunks(uniq, BATCH):
            lines = []
            for i, (a, b) in enumerate(chunk):
                lines += [f"g{i + 1}:", "  " + self._describe(f"m{a['id']}", a), "  " + self._describe(f"m{b['id']}", b)]
            out = self._call(SETTLE_PROMPT, "\n".join(lines))
            if not out:
                continue
            by_group = {f"g{i + 1}": p for i, p in enumerate(chunk)}
            for d in out.get("decisions", []):
                pair = by_group.get(str(d.get("group")))
                if not pair:
                    continue
                a, b = pair
                if str(d.get("a")) == f"m{b['id']}":
                    a, b = b, a
                result = self._pair(a, b, str(d.get("decision")), now, str(d.get("why", "")))
                self.report["settled"].append({"a": a["id"], "b": b["id"], "decision": d.get("decision"),
                                               "result": result})

    # D4

    def close_stale(self, now: str) -> None:
        items = [m for m in self.store.current(now)
                 if m["kind"] == "todo" or (m["kind"] == "event" and (m["event_at"] or "") >= now)]
        everything = self.store.memories()
        for chunk in _chunks(items, CLOSE_BATCH):
            lines, ev_map = [], {}
            for i, m in enumerate(chunk):
                lines.append(self._describe(f"t{i + 1}", m))
                evidence = self._evidence(m, everything)
                if not evidence:
                    lines.append("  (no later evidence)")
                for _label, kind, ref, said, text in evidence:
                    key = f"e{len(ev_map) + 1}"
                    ev_map[key] = (kind, ref, said, text, f"t{i + 1}")
                    lines.append(f"  {key} · {said[:16]} · {text}")
            if not ev_map:
                continue
            out = self._call(CLOSE_PROMPT, "\n".join(lines))
            if not out:
                continue
            items_by = {f"t{i + 1}": m for i, m in enumerate(chunk)}
            for c in out.get("closes", []):
                m = items_by.get(str(c.get("item")))
                ev = ev_map.get(str(c.get("evidence")))
                if not m or not ev or ev[4] != str(c.get("item")):
                    continue
                kind, ref, said, text, _ = ev
                if said <= m["valid_from"]:
                    continue
                self.store.end_memory(m["id"], said, "ended")
                sid = ref if kind == "source" else None
                if sid is None:
                    srcs = self.store.memory_sources(ref)
                    sid = srcs[0]["id"] if srcs else None
                if sid is not None:
                    self.store.attach_source(m["id"], sid, c.get("quote"), "ended")
                self.store.log(now, "dream", "applied", op="close", target_id=m["id"], reason=c.get("why"))
                self.report["closed"].append({"id": m["id"], "text": m["text"], "by": text, "why": c.get("why")})

    def _evidence(self, m: dict[str, Any], everything: list[dict[str, Any]]):
        later = [x for x in everything if x["valid_from"] > m["valid_from"] and x["id"] != m["id"]]
        later_ids = {x["id"] for x in later}
        by_id = {x["id"]: x for x in later}
        picked: list[int] = []
        for i, _ in self.store.vector_top(self.store.vector(m["id"]), list(later_ids), EVIDENCE):
            picked.append(i)
        for i in self.store.bm25_memories(m["text"], EVIDENCE, later_ids):
            if i not in picked:
                picked.append(i)
        out = [(f"m{i}", "memory", i, by_id[i]["valid_from"], f"memory: {by_id[i]['text']}{self._quote_line(by_id[i])}")
               for i in picked]
        own = {s["id"] for s in self.store.memory_sources(m["id"])}
        for s in self.store.bm25_sources(m["text"], EVIDENCE,
                                         "s.skip_reason IS NULL AND s.said_at > ?", (m["valid_from"],)):
            if s["id"] not in own:
                out.append((f"s{s['id']}", "source", s["id"], s["said_at"],
                            f"message from {s['author']} in {s['thread_title']}: \"{s['text']}\""))
        return out

    # D5

    def merge(self, now: str) -> None:
        cur = self.store.current(now)
        if len(cur) < 2:
            return
        ids = [m["id"] for m in cur]
        mat = np.stack([self.store.vector(i) for i in ids])
        sims = mat @ mat.T
        linked = {(r["from_id"], r["to_id"]) for r in self.store.db.execute("SELECT from_id, to_id FROM links")}
        by_id = {m["id"]: m for m in cur}
        pairs = []
        for i, j in zip(*np.where(np.triu(sims, 1) > MERGE_COSINE), strict=True):
            a, b = ids[i], ids[j]
            if (a, b) in linked or (b, a) in linked:
                continue
            ea, eb = by_id[a]["event_at"], by_id[b]["event_at"]
            if ea and eb and ea[:10] != eb[:10]:
                continue
            pairs.append((by_id[a], by_id[b]))
        for chunk in _chunks(pairs, BATCH):
            lines = []
            for k, (a, b) in enumerate(chunk):
                lines += [f"p{k + 1}:", "  " + self._describe("a", a), "  " + self._describe("b", b)]
            out = self._call(MERGE_PROMPT, "\n".join(lines))
            if not out:
                continue
            for d in out.get("pairs", []):
                try:
                    a, b = chunk[int(str(d.get("pair", "p0"))[1:]) - 1]
                except (ValueError, IndexError):
                    continue
                result = self._pair(a, b, str(d.get("decision")), now, str(d.get("why", "")))
                self.report["merged"].append({"a": a["id"], "b": b["id"], "decision": d.get("decision"),
                                              "result": result})


def _chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def dumps(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=1, default=str)
