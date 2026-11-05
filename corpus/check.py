"""Check a merged scenario C corpus against the frozen spec.

    python -m corpus.check corpus/scenario_c.json

Passes only when every planted id appears exactly once with its exact text,
sender and time; every message is inside Nov 2 00:00 to Nov 26 20:00; noise
targets are met within 10 percent; no noise message breaks a noise rule;
each conversation is in time order; and every noise sender is a member of its
conversation. Prints every failure with its label and exits 1 on any failure.

The rule functions here are also what `corpus.noise` uses to drop messages.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from corpus.scenario_c import BY_KEY, NOW, PLANTED_PATH, START, read_json
from sme.store import repo

BANNED_TERMS: tuple[str, ...] = (
    "dinner", "brunch", "lunch", "breakfast", "deck", "cap table", "landlord", "Kapoor",
    "plumber", "tap", "charger", "photo", "Diwali", "train", "ticket", "offsite", "form",
    "Q4", "flight", "Delhi", "conference", "6E", "vegetarian", "veg", "fish", "meat",
    "kebab", "owe", "pay back", "1,200", "Arjun", "restaurant", "table", "Friday",
    "Saturday", "Sunday",
)  # fmt: skip

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
YOU_BANNED_TERMS: tuple[str, ...] = (
    "I'll", "I will", "will do", "I can", "promise", "by tomorrow",
    *(f"by {d}" for d in WEEKDAYS),
    "see you",
)  # fmt: skip

PLANTED_LABEL = re.compile(r"^c\d{2}$")
NOISE_LABEL = re.compile(r"^n\d{4,}$")
TOLERANCE = 0.10


def term_pattern(term: str) -> re.Pattern[str]:
    """Whole word or phrase, any case, with an optional trailing s."""
    words = [re.escape(w) for w in term.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"s?(?!\w)", re.IGNORECASE)


BANNED_PATTERNS = [(t, term_pattern(t)) for t in BANNED_TERMS]
YOU_BANNED_PATTERNS = [(t, term_pattern(t)) for t in YOU_BANNED_TERMS]


def _normalize(text: str) -> str:
    # Curly apostrophes count as straight ones, so "I’ll" matches "I'll".
    return text.replace("’", "'").replace("‘", "'")


def banned_hits(body: str) -> list[str]:
    text = _normalize(body)
    return [t for t, p in BANNED_PATTERNS if p.search(text)]


def you_banned_hits(body: str) -> list[str]:
    text = _normalize(body)
    return [t for t, p in YOU_BANNED_PATTERNS if p.search(text)]


def noise_rule_violations(sender: str, body: str, members: tuple[str, ...]) -> list[str]:
    """Why a noise message breaks the noise rules. Empty means it passes."""
    out = [f"banned term {t!r}" for t in banned_hits(body)]
    if sender == repo.SELF_LABEL:
        out += [f"You message has banned term {t!r}" for t in you_banned_hits(body)]
    if sender not in members:
        out.append(f"sender {sender!r} is not a member ({', '.join(members)})")
    return out


def target_range(target: int) -> tuple[int, int]:
    """Smallest and largest noise count within 10 percent of the target."""
    lo = target - int(target * TOLERANCE)
    hi = target + int(target * TOLERANCE)
    return lo, hi


def check(scenario: dict[str, Any], planted: dict[str, Any]) -> list[str]:
    """Return every failure as a line that starts with the message label (or '-')."""
    failures: list[str] = []
    messages = scenario.get("messages", [])
    expected = {m["label"]: m for m in planted["messages"]}

    # Planted ids: each exactly once, with every field as written.
    seen = Counter(m.get("label") for m in messages)
    for label in expected:
        if seen[label] == 0:
            failures.append(f"{label}: planted message is missing")
        elif seen[label] > 1:
            failures.append(f"{label}: planted message appears {seen[label]} times")
    for msg in messages:
        label = msg.get("label")
        if label in expected:
            for field, value in expected[label].items():
                if msg.get(field) != value:
                    failures.append(f"{label}: {field} is {msg.get(field)!r}, expected {value!r}")
        elif not (isinstance(label, str) and NOISE_LABEL.match(label)):
            failures.append(f"{label}: label is neither a planted id nor a noise label")

    # Time range.
    start, end = repo.parse_time(START), repo.parse_time(NOW)
    for msg in messages:
        label = msg.get("label")
        try:
            sent = repo.parse_time(msg["sent_at"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{label}: sent_at {msg.get('sent_at')!r} is not an ISO time")
            continue
        if not start <= sent <= end:
            failures.append(f"{label}: sent_at {msg['sent_at']} is outside {START} to {NOW}")

    # Noise rules and senders.
    noise_count: Counter[str] = Counter()
    for msg in messages:
        label = msg.get("label")
        if not (isinstance(label, str) and NOISE_LABEL.match(label)):
            continue
        conv = BY_KEY.get(msg.get("thread", ""))
        if conv is None:
            failures.append(f"{label}: unknown conversation {msg.get('thread')!r}")
            continue
        noise_count[conv.key] += 1
        for reason in noise_rule_violations(msg["sender"], msg["body"], conv.members):
            failures.append(f"{label}: {reason}: {msg['body']!r}")

    # Noise targets.
    for conv in BY_KEY.values():
        lo, hi = target_range(conv.noise_target)
        n = noise_count[conv.key]
        if not lo <= n <= hi:
            failures.append(
                f"-: {conv.key} has {n} noise messages, target {conv.noise_target}"
                f" allows {lo} to {hi}"
            )

    # Time order within each conversation, in file order.
    last: dict[str, tuple[Any, str]] = {}
    by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for msg in messages:
        by_thread[msg.get("thread", "")].append(msg)
    for thread, msgs in by_thread.items():
        for msg in msgs:
            try:
                sent = repo.parse_time(msg["sent_at"])
            except (KeyError, TypeError, ValueError):
                continue
            prev = last.get(thread)
            if prev and sent < prev[0]:
                failures.append(
                    f"{msg.get('label')}: {thread} is out of time order,"
                    f" {msg['sent_at']} comes after {prev[1]}"
                )
            last[thread] = (sent, msg.get("label"))

    return failures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check a merged scenario C corpus")
    parser.add_argument("scenario", help="merged corpus, e.g. corpus/scenario_c.json")
    parser.add_argument("--planted", default=str(PLANTED_PATH), help="planted messages JSON")
    args = parser.parse_args(argv)

    scenario = read_json(args.scenario)
    failures = check(scenario, read_json(args.planted))
    for line in failures:
        print(line)
    total = len(scenario.get("messages", []))
    if failures:
        print(f"FAIL: {len(failures)} failures in {Path(args.scenario)} ({total} messages)")
        sys.exit(1)
    print(f"OK: {Path(args.scenario)} passes ({total} messages)")


if __name__ == "__main__":
    main()
