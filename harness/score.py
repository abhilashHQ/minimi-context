"""End-state checks on the planted scenario C facts, the pilot's 19 checks over the store."""

from __future__ import annotations

from typing import Any

from harness.store import Store
from harness.text import overlap


def checks(store: Store, now: str) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    cur = store.current(now)

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
    out: dict[str, bool] = {}
    for name, p in expected_open.items():
        out[f"open: {name}"] = any(p(m) for m in open_todos)
    for name, p in expected_closed.items():
        out[f"closed: {name}"] = not any(p(m) for m in open_todos)
    matched = [m for m in open_todos if any(p(m) for p in expected_open.values())]
    out["no extra open todos"] = len(open_todos) == len(matched)
    out["dinner Nov 27 at 9:30 PM"] = has(lambda m: m["kind"] == "event" and at(m, "2026-11-27T21:30"))
    out["no dinner Nov 27 at 8 PM"] = not has(lambda m: m["kind"] == "event" and at(m, "2026-11-27T20:00"))
    out["dinner Nov 20 at 9:30, not 8"] = has(lambda m: at(m, "2026-11-20T21:30")) and not has(
        lambda m: m["kind"] == "event" and at(m, "2026-11-20T20:00"))
    out["brunch Nov 28 at 9:30 AM"] = has(lambda m: m["kind"] == "event" and at(m, "2026-11-28T09:30"))
    out["lunch Nov 29 at 1 PM"] = has(lambda m: m["kind"] == "event" and at(m, "2026-11-29T13:00"))
    out["Sam eats fish now"] = has(lambda m: "fish" in txt(m))
    out["Sam not currently vegetarian"] = not has(lambda m: "vegetarian" in txt(m) and "fish" not in txt(m)
                                                  and overlap(m["people"], ["Sam"]))
    out["flight not given to Sam"] = not has(lambda m: "6e 2134" in txt(m) and overlap(m["people"], ["Sam"]))
    out["flight number kept, on Priya"] = has(lambda m: "6e 2134" in txt(m) and overlap(m["people"], ["Priya"]))
    return out, open_todos
