"""Harness CLI.

    python -m harness.cli replay --corpus corpus/scenario_d.json --db runs/harness/d.db
    python -m harness.cli add --db runs/harness/d.db --thread sam --msg "Sam: can we push to 10?" --msg "You: sure"
    python -m harness.cli ask --db runs/harness/d.db "What time is dinner with Sam tomorrow?"
    python -m harness.cli dream --db runs/harness/d.db
    python -m harness.cli score --db runs/harness/d.db
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from typing import Any

from harness.ingest import Bursts, arrival_order, ingest, load_corpus, to_source
from harness.llm import LLM
from harness.store import Store
from harness.text import Embedder, dt, iso
from harness.write import Writer


class Engine:
    def __init__(self, db: str, model: str | None = None) -> None:
        self.store = Store(db)
        self.llm = LLM(self.store, **({"model": model} if model else {}))
        self.embedder = Embedder()
        self.bursts = Bursts(self.store)
        self.writer = Writer(self.store, self.llm, self.embedder)
        self.reports: list[dict[str, Any]] = []
        self.verbose = True

    def drain(self) -> None:
        for bid in self.bursts.ready():
            rep = self.writer.process(bid)
            self.reports.append(rep)
            if self.verbose:
                ops = ",".join(
                    f"{o['op']}{'' if o['outcome'] == 'applied' else '!'}" for o in rep["ops"]
                ) or ("ERROR" if rep.get("error") else "-")
                s = self.llm.stats
                print(f"[{len(self.reports):4}] {rep['thread']:16} {','.join(map(str, rep['labels']))[:40]:40}"
                      f" {ops[:60]}  calls={s.calls} hits={s.cache_hits}", flush=True)
        self.store.commit()

    def feed(self, msg: dict[str, Any], titles: dict[str, str]) -> str:
        src = to_source(msg, titles)
        self.bursts.tick(src["seen_at"])
        self.drain()
        _, outcome = ingest(self.store, self.bursts, src)
        return outcome


def cmd_replay(args) -> None:
    eng = Engine(args.db, args.model)
    corpus = load_corpus(args.corpus)
    titles = {c["key"]: c["title"] for c in corpus["conversations"]}
    eng.store.set_meta("mode", "replay")
    eng.store.set_meta("corpus", args.corpus)
    eng.store.set_meta("titles", json.dumps(titles))
    msgs = arrival_order(corpus)
    if args.planted:
        msgs = [m for m in msgs if str(m.get("label", "")).startswith("c")]
    if args.until:
        msgs = [m for m in msgs if (m.get("seen_at") or m["sent_at"]) <= args.until]
    t0 = time.time()
    counts: dict[str, int] = {}
    for m in msgs:
        outcome = eng.feed(m, titles)
        counts[outcome] = counts.get(outcome, 0) + 1
    now = args.until or corpus["now"]
    eng.bursts.close_all(now)
    eng.drain()
    eng.store.set_meta("now", now)
    eng.store.commit()
    if args.dream:
        from harness.dream import Dreamer

        Dreamer(eng.store, eng.llm, eng.embedder, eng.writer).run(now)
    s = eng.llm.stats
    summary = {
        "sources": counts, "bursts": len(eng.reports), "calls": s.calls, "cache_hits": s.cache_hits,
        "tokens_in": s.tokens_in, "tokens_out": s.tokens_out, "llm_seconds": round(s.seconds),
        "wall_seconds": round(time.time() - t0), "failed": [r for r in eng.reports if r.get("error")],
        "ops": op_counts(eng.store),
    }
    print(json.dumps(summary, indent=1, default=str))


def op_counts(store: Store) -> dict[str, int]:
    rows = store.db.execute("SELECT stage, op, outcome, count(*) n FROM ops_log GROUP BY 1, 2, 3")
    return {f"{r['stage']}:{r['op']}:{r['outcome']}": r["n"] for r in rows}


def cmd_add(args) -> None:
    eng = Engine(args.db, args.model)
    store = eng.store
    titles = json.loads(store.get_meta("titles", "{}"))
    at = dt(args.at) if args.at else dt(store.get_meta("now")) + timedelta(minutes=1)
    app = args.app or store.db.execute(
        "SELECT app FROM sources WHERE thread_key = ? ORDER BY said_at DESC LIMIT 1", (args.thread,)
    ).fetchone()[0]
    for i, line in enumerate(args.msg):
        sender, _, body = line.partition(":")
        stamp = iso(at + timedelta(minutes=i))
        msg = {"label": f"live{int(time.time())}{i}", "app": app, "thread": args.thread,
               "sender": sender.strip(), "sent_at": stamp, "seen_at": stamp, "body": body.strip()}
        print(f"{msg['sender']}: {msg['body']} -> {eng.feed(msg, titles)}")
    store.set_meta("now", stamp)
    if args.flush:
        eng.bursts.close_all(stamp)
        eng.drain()
    store.commit()


def cmd_ask(args) -> None:
    from harness.retrieve import Retriever

    store = Store(args.db)
    r = Retriever(store, Embedder())
    print(r.get_context(args.question))


def cmd_dream(args) -> None:
    from harness.dream import Dreamer

    eng = Engine(args.db, args.model)
    now = args.now or eng.store.get_meta("now")
    print(json.dumps(Dreamer(eng.store, eng.llm, eng.embedder, eng.writer).run(now), indent=1))


def cmd_score(args) -> None:
    from harness.score import checks

    store = Store(args.db)
    out, open_todos = checks(store, store.get_meta("now"))
    for k, v in out.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(f"{sum(out.values())} of {len(out)}")
    print("open todos:")
    for m in open_todos:
        print(f"  m{m['id']} {m['event_at'] or '-'} {m['text']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay")
    r.add_argument("--corpus", required=True)
    r.add_argument("--db", required=True)
    r.add_argument("--model")
    r.add_argument("--planted", action="store_true", help="only the planted c-labelled messages")
    r.add_argument("--until", help="stop at this arrival time and treat it as now")
    r.add_argument("--no-dream", dest="dream", action="store_false")
    r.set_defaults(fn=cmd_replay)

    a = sub.add_parser("add")
    a.add_argument("--db", required=True)
    a.add_argument("--thread", required=True)
    a.add_argument("--msg", action="append", required=True, help='"Sender: text", repeatable')
    a.add_argument("--app")
    a.add_argument("--at", help="said_at of the first message (default: now + 1 minute)")
    a.add_argument("--no-flush", dest="flush", action="store_false")
    a.add_argument("--model")
    a.set_defaults(fn=cmd_add)

    q = sub.add_parser("ask")
    q.add_argument("--db", required=True)
    q.add_argument("question")
    q.set_defaults(fn=cmd_ask)

    d = sub.add_parser("dream")
    d.add_argument("--db", required=True)
    d.add_argument("--now")
    d.add_argument("--model")
    d.set_defaults(fn=cmd_dream)

    s = sub.add_parser("score")
    s.add_argument("--db", required=True)
    s.set_defaults(fn=cmd_score)

    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    args.fn(args)


if __name__ == "__main__":
    main()
