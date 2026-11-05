"""Write-path pilot: run design-final's W3 to W8 over the 48 planted scenario C messages
with one Bedrock model, then score the end state against the known answers.

    .venv/Scripts/python.exe -m pilot.write_pilot --model openai.gpt-oss-120b-1:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import boto3
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sentence_transformers import SentenceTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "scenario_c" / "planted.json"
STOP = set("a an the to of and or for in on at by is are i i'll you your me my we it this that be with can will yes so".split())
HONORIFICS = {"mr.", "mrs.", "ms.", "dr.", "mr", "mrs", "ms", "dr"}

PROMPT = """You keep a personal memory for one user, called "You". You get a chunk of new messages from one conversation, a few earlier messages for context, and existing memories that might be related. Decide what to write. Reply with JSON only.

A memory is one self-contained sentence from the user's point of view with absolute dates, like "You'll send Priya the deck by Fri Nov 13, 6:00 PM" or "Dinner with Sam on Fri Nov 20 at 9:30 PM".

Kinds: event (something planned at a time), todo (something someone agreed to do), fact, preference, status (true for a while), episode (something that happened).
owner, for todos only: "you" if the user has to act, "other" if someone else has to.

Ops:
- new: nothing in the candidates covers it. If a candidate shows it is already done or no longer true, add closed_by with that candidate's id.
- same: says what candidate cX already says. A copy of an ended candidate stays on that ended candidate.
- replaces: candidate cX is no longer true; give the new version.
- extends: adds detail to cX; both stay true.
- ends: cX is done, cancelled, or stopped being true, with no replacement.

Rules:
- Every op cites one message as msg (b1, b2, ...) and a quote copied word for word from that message.
- Resolve relative dates ("Friday", "tomorrow", "next Friday") against the cited message's own date, never today. Use the calendar given with the messages to look up weekdays; don't compute them yourself.
- A time without AM or PM takes the obvious meaning from context: dinner, drinks and "tonight" are evening, breakfast and brunch are morning.
- An event that simply happened is not ended. Only end an event that was cancelled or replaced.
- Keep exact identifiers (flight numbers, amounts, file names) in the memory text.
- A request becomes a todo for you only when you agree to it. A proposal ("can we do 9:30?") is not an agreement until someone accepts it.
- "Same as last time" and similar phrases: work out the real value from the candidates.
- Skip small talk unless it closes or changes something.
- Give the end state of the chunk: if something changed twice inside the chunk, report only the final version.

JSON shape:
{"ops": [{"op": "new|same|replaces|extends|ends", "target": "c3 or null", "msg": "b1", "quote": "...", "text": "...", "kind": "...", "owner": "you|other|null", "people": ["..."], "event_at": "YYYY-MM-DDTHH:MM or null", "closed_by": "c5 or null"}]}
For same and ends, text, kind, people and event_at may be null.
"""


def day(ts: str) -> str:
    return datetime.fromisoformat(ts).strftime("%a %Y-%m-%d %H:%M")


def tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1}


def name_tokens(name: str) -> list[str]:
    return [t for t in name.lower().split() if t not in HONORIFICS]


def same_person(a: str, b: str) -> bool:
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb or "you" in (ta[0], tb[0]):
        return False
    return ta[0].startswith(tb[0]) or tb[0].startswith(ta[0])


def overlap(people_a, people_b) -> bool:
    return any(same_person(x, y) for x in people_a or [] for y in people_b or [])


def feed(corpus):
    order = [c["key"] for c in corpus["conversations"]]
    titles = {c["key"]: (c["app"], c["title"]) for c in corpus["conversations"]}
    by_thread = {}
    for m in corpus["messages"]:
        by_thread.setdefault(m["thread"], []).append(m)
    bursts = []
    for key in order:
        msgs = sorted(by_thread.get(key, []), key=lambda m: m["sent_at"])
        cur = []
        for m in msgs:
            t = datetime.fromisoformat(m["sent_at"])
            if cur:
                last = datetime.fromisoformat(cur[-1]["sent_at"])
                first = datetime.fromisoformat(cur[0]["sent_at"])
                if (t - last).total_seconds() > 3600 or (t - first).total_seconds() > 4 * 3600 or len(cur) >= 20:
                    bursts.append((key, cur))
                    cur = []
            cur.append(m)
        if cur:
            bursts.append((key, cur))
    return bursts, by_thread, titles


def chronological(bursts):
    """Live order: a burst is processed once its last message has arrived."""
    return sorted(bursts, key=lambda kb: kb[1][-1]["sent_at"])


class Pilot:
    def __init__(self, model: str, region: str):
        self.model = model
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.mem: list[dict] = []
        self.log: list[dict] = []
        self.tokens_in = self.tokens_out = 0
        self.calls = 0
        self.seconds = 0.0

    def embed(self, text):
        return self.embedder.encode([text], normalize_embeddings=True)[0]

    def current(self):
        return [m for m in self.mem if m["valid_to"] is None]

    def candidates(self, key, burst, participants):
        picked: list[dict] = []

        def add(ms, cap):
            n = 0
            for m in ms:
                if len(picked) >= 40 or n >= cap:
                    return
                if m not in picked:
                    picked.append(m)
                    n += 1

        thread = [m for m in self.mem if key in m["threads"]]
        add(sorted(thread, key=lambda m: m["valid_from"], reverse=True), 10)
        start = burst[0]["sent_at"]
        ppl = [m for m in self.mem if overlap(m["people"], participants)
               and ((m["valid_to"] is None and m["kind"] in ("todo", "event", "status")) or m["valid_from"] > start)]
        ppl.sort(key=lambda m: (m["kind"] != "todo", m["valid_from"]))
        add(ppl, 15)
        cur = self.current()
        if cur:
            mat = np.stack([m["emb"] for m in cur])
            for msg in burst:
                sims = mat @ self.embed(msg["body"])
                add([cur[i] for i in np.argsort(-sims)[:3]], 3)
                tk = tokens(msg["body"])
                kw = sorted(cur, key=lambda m: -len(tk & tokens(m["text"])))
                add([m for m in kw[:3] if tk & tokens(m["text"])], 3)
        return picked

    def call(self, user):
        kwargs = dict(modelId=self.model, system=[{"text": PROMPT}],
                      messages=[{"role": "user", "content": [{"text": user}]}],
                      inferenceConfig={"maxTokens": 4000, "temperature": 0})
        if "gpt-oss" in self.model:
            kwargs["additionalModelRequestFields"] = {"reasoning_effort": "medium"}
        t0 = time.time()
        r = self.client.converse(**kwargs)
        self.seconds += time.time() - t0
        self.calls += 1
        self.tokens_in += r["usage"]["inputTokens"]
        self.tokens_out += r["usage"]["outputTokens"]
        text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
        a, b = text.find("{"), text.rfind("}")
        return json.loads(text[a:b + 1]), text

    def run_burst(self, key, burst, thread_msgs, title):
        app, ttl = title
        senders = {m["sender"] for m in burst if m["sender"] != "You"}
        participants = list(senders | ({ttl} if app == "whatsapp_web" and key not in ("flat-4b", "family", "college") else set()))
        cands = self.candidates(key, burst, participants)
        cid = {f"c{i + 1}": m for i, m in enumerate(cands)}
        bid = {f"b{i + 1}": m for i, m in enumerate(burst)}
        idx = thread_msgs.index(burst[0])
        ctx = thread_msgs[max(0, idx - 3):idx]
        lines = [f"Conversation: {ttl} ({'WhatsApp' if app == 'whatsapp_web' else 'Gmail'})", "", "Earlier messages (context only):"]
        lines += [f"- {day(m['sent_at'])} {m['sender']}: {m['body']}" for m in ctx] or ["- none"]
        d0 = datetime.fromisoformat(burst[0]["sent_at"]).date()
        cal = [(d0.fromordinal(d0.toordinal() + i)) for i in range(-7, 15)]
        lines += ["", "Calendar: " + ", ".join(d.strftime("%a %Y-%m-%d") for d in cal)]
        lines += ["", "New messages:"]
        lines += [f"{b} · {day(m['sent_at'])} · {m['sender']}: {m['body']}" for b, m in bid.items()]
        lines += ["", "Existing memories:"]
        for c, m in cid.items():
            status = "current" if m["valid_to"] is None else f"ENDED ({m['end_reason']})"
            due = f" · at {m['event_at']}" if m["event_at"] else ""
            own = f" ({m['owner']})" if m["owner"] else ""
            lines.append(f"{c} [{status}] {m['kind']}{own}{due} · people: {', '.join(m['people']) or '-'} · said {m['valid_from'][:16]} · {m['text']}")
        if not cid:
            lines.append("none")
        user = "\n".join(lines)
        try:
            out, raw = self.call(user)
        except (json.JSONDecodeError, ValueError):
            try:
                out, raw = self.call(user + "\n\nReply with valid JSON only.")
            except (json.JSONDecodeError, ValueError) as e:
                self.log.append({"thread": key, "burst": [m["label"] for m in burst], "error": f"parse: {e}"})
                return
        entry = {"thread": key, "burst": [m["label"] for m in burst], "candidates": {c: m["id"] for c, m in cid.items()},
                 "ops": out.get("ops", []), "results": []}
        ops = sorted(out.get("ops", []), key=lambda o: bid.get(o.get("msg"), burst[-1])["sent_at"])
        for op in ops:
            entry["results"].append(self.apply(op, cid, bid, key))
        self.log.append(entry)

    def apply(self, op, cid, bid, key):
        kind = op.get("op")
        msg = bid.get(op.get("msg"))
        if not msg:
            return {"op": kind, "outcome": "rejected", "reason": "bad msg"}
        q = (op.get("quote") or "").lower().strip()
        body = msg["body"].lower()
        if not q or (q not in body and SequenceMatcher(None, q, body).find_longest_match(0, len(q), 0, len(body)).size < 0.9 * len(q)):
            return {"op": kind, "outcome": "rejected", "reason": "quote not in message"}
        target = cid.get(op.get("target")) if op.get("target") else None
        if kind in ("same", "replaces", "extends", "ends") and not target:
            return {"op": kind, "outcome": "rejected", "reason": "target not a candidate"}
        said = msg["sent_at"]
        if kind in ("same", "replaces") and target["people"] and op.get("people") and not overlap(target["people"], op["people"]):
            kind = "new"
            target = None
        if kind == "same":
            target["sources"].append(msg["label"])
            target["threads"].add(key)
            return {"op": "same", "outcome": "applied", "target": target["id"]}
        if kind == "ends":
            if said <= target["valid_from"]:
                return {"op": kind, "outcome": "rejected", "reason": "ends is older than target"}
            target["valid_to"], target["end_reason"] = said, "ended"
            target["ended_by"] = msg["label"]
            return {"op": "ends", "outcome": "applied", "target": target["id"]}
        if not op.get("text"):
            return {"op": kind, "outcome": "rejected", "reason": "no text"}
        new = {"id": f"m{len(self.mem) + 1}", "text": op["text"], "kind": op.get("kind") or "fact",
               "owner": op.get("owner") if op.get("owner") in ("you", "other") else None,
               "people": [p for p in (op.get("people") or []) if p and p.lower() != "you"],
               "event_at": op.get("event_at"), "valid_from": said, "valid_to": None, "end_reason": None,
               "sources": [msg["label"]], "threads": {key}, "emb": self.embed(op["text"])}
        outcome = {"op": kind, "outcome": "applied", "id": new["id"]}
        if kind == "replaces":
            if said > target["valid_from"]:
                target["valid_to"], target["end_reason"] = said, "replaced"
            else:
                new["valid_to"], new["end_reason"] = target["valid_from"], "replaced"
                outcome["note"] = "older than target, inserted ended"
        closer = cid.get(op.get("closed_by")) if op.get("closed_by") else None
        if kind == "new" and closer:
            if closer["valid_from"] > said:
                new["valid_to"], new["end_reason"] = closer["valid_from"], "ended"
                outcome["note"] = f"closed_by {closer['id']}"
            else:
                outcome["note"] = "closed_by rejected, closer is older"
        self.mem.append(new)
        return outcome


def score(mem):
    cur = [m for m in mem if m["valid_to"] is None]

    def has(pred):
        return any(pred(m) for m in cur)

    def txt(m):
        return m["text"].lower()

    def at(m, prefix):
        return (m["event_at"] or "").startswith(prefix)

    open_todos = [m for m in cur if m["kind"] == "todo" and m["owner"] != "other"]
    expected_open = {
        "offsite form": lambda m: "offsite" in txt(m),
        "cap table": lambda m: "cap table" in txt(m),
        "train tickets": lambda m: "train" in txt(m) or "ticket" in txt(m),
        "pay Rohan back": lambda m: "1,200" in txt(m) or "1200" in txt(m) or ("rohan" in txt(m) and "pay" in txt(m)),
        "Diwali photos": lambda m: "diwali" in txt(m) or "photo" in txt(m),
    }
    expected_closed = {
        "deck": lambda m: "deck" in txt(m),
        "Q4 numbers": lambda m: "q4" in txt(m),
        "landlord call": lambda m: "landlord" in txt(m) or "kapoor" in txt(m),
        "charger": lambda m: "charger" in txt(m),
    }
    checks = {}
    for name, p in expected_open.items():
        checks[f"open: {name}"] = any(p(m) for m in open_todos)
    for name, p in expected_closed.items():
        checks[f"closed: {name}"] = not any(p(m) for m in open_todos)
    matched = [m for m in open_todos if any(p(m) for p in expected_open.values())]
    checks["no extra open todos"] = len(open_todos) == len(matched)
    checks["dinner Nov 27 at 9:30 PM"] = has(lambda m: m["kind"] == "event" and at(m, "2026-11-27T21:30"))
    checks["no dinner Nov 27 at 8 PM"] = not has(lambda m: m["kind"] == "event" and at(m, "2026-11-27T20:00"))
    checks["dinner Nov 20 at 9:30, not 8"] = has(lambda m: at(m, "2026-11-20T21:30")) and not has(lambda m: at(m, "2026-11-20T20:00"))
    checks["brunch Nov 28 at 9:30 AM"] = has(lambda m: m["kind"] == "event" and at(m, "2026-11-28T09:30"))
    checks["lunch Nov 29 at 1 PM"] = has(lambda m: m["kind"] == "event" and at(m, "2026-11-29T13:00"))
    checks["Sam eats fish now"] = has(lambda m: "fish" in txt(m))
    checks["Sam not currently vegetarian"] = not has(lambda m: "vegetarian" in txt(m) and "fish" not in txt(m))
    checks["flight not given to Sam"] = not has(lambda m: "6e 2134" in txt(m) and overlap(m["people"], ["Sam"]))
    checks["flight number kept, on Priya"] = has(lambda m: "6e 2134" in txt(m) and overlap(m["people"], ["Priya"]))
    return checks, open_todos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--out", default=str(ROOT / "runs" / "pilot"))
    ap.add_argument("--order", choices=["corpus", "chrono"], default="corpus")
    args = ap.parse_args()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    bursts, by_thread, titles = feed(corpus)
    if args.order == "chrono":
        bursts = chronological(bursts)
    for k in by_thread:
        by_thread[k].sort(key=lambda m: m["sent_at"])
    p = Pilot(args.model, args.region)
    for key, burst in bursts:
        p.run_burst(key, burst, by_thread[key], titles[key])
        print(f"{key:14} {','.join(m['label'] for m in burst):20} ops={len(p.log[-1].get('ops', []))}", flush=True)
    checks, open_todos = score(p.mem)
    outs = [r for e in p.log for r in e.get("results", [])]
    summary = {
        "model": args.model, "bursts": len(bursts), "calls": p.calls,
        "tokens_in": p.tokens_in, "tokens_out": p.tokens_out, "seconds": round(p.seconds, 1),
        "ops_applied": sum(r["outcome"] == "applied" for r in outs),
        "ops_rejected": [r for r in outs if r["outcome"] == "rejected"],
        "parse_errors": [e for e in p.log if "error" in e],
        "checks_passed": f"{sum(checks.values())} of {len(checks)}", "checks": checks,
        "open_todos": [m["text"] for m in open_todos],
    }
    out = Path(args.out) / (args.model.replace(":", "_") + ("_chrono" if args.order == "chrono" else ""))
    out.mkdir(parents=True, exist_ok=True)
    mem = [{k: (sorted(v) if isinstance(v, set) else v) for k, v in m.items() if k != "emb"} for m in p.mem]
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "memories.json").write_text(json.dumps(mem, indent=1), encoding="utf-8")
    (out / "log.json").write_text(json.dumps(p.log, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("ops_rejected",)}, indent=1))
    print("rejected:", len(summary["ops_rejected"]))


if __name__ == "__main__":
    main()
