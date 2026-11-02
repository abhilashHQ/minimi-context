"""Scenario C: conversation table and the merge step.

    python -m corpus.scenario_c build

Merges the hand-written planted messages (`corpus/scenario_c/planted.json`)
and the frozen generated noise (`corpus/scenario_c/noise.json`) into
`corpus/scenario_c.json`, in the format `repo.load_scenario` reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from sme.store import repo

SCENARIO = "c"
NOW = "2026-11-26T20:00:00+05:30"
START = "2026-11-02T00:00:00+05:30"
FIRST_DAY = date(2026, 11, 2)
LAST_DAY = date(2026, 11, 26)

CORPUS_DIR = Path(__file__).resolve().parent
PLANTED_PATH = CORPUS_DIR / "scenario_c" / "planted.json"
NOISE_PATH = CORPUS_DIR / "scenario_c" / "noise.json"
OUT_PATH = CORPUS_DIR / "scenario_c.json"

DATE_SOURCE = {"gmail": "absolute", "whatsapp_web": "separator"}
DESCRIPTION = (
    "Fictional test corpus. All times are Asia/Kolkata (IST, +05:30). Conversations are"
    " listed in ingest order and messages within a conversation in time order. Labels"
    " c01 to c48 are planted; n0001 and up are generated noise."
)


@dataclass(frozen=True)
class Conversation:
    key: str
    app: str
    title: str
    # Noise senders allowed in this conversation, You first.
    members: tuple[str, ...]
    noise_target: int
    # Fixed schedule: one message from members[1] at this HH:MM every weekday.
    weekday_time: str | None = None


# Ingest order: Gmail first, then WhatsApp.
CONVERSATIONS: tuple[Conversation, ...] = (
    Conversation("friday-dinner", "gmail", "Friday dinner", ("You", "Samuel Rao"), 0),
    Conversation("investor-deck", "gmail", "Investor deck", ("You", "Priya Nair"), 0),
    Conversation("q4-numbers", "gmail", "Q4 numbers", ("You", "Dev Mehta"), 0),
    Conversation("offsite", "gmail", "Team offsite", ("You", "Dev Mehta"), 0),
    Conversation("standup", "gmail", "Standup notes", ("You", "Dev Mehta"), 18, "09:30"),
    Conversation("newsletter", "gmail", "The Weekly Byte", ("You", "The Weekly Byte"), 18, "07:00"),
    Conversation("sam", "whatsapp_web", "Sam", ("You", "Sam"), 300),
    Conversation("priya", "whatsapp_web", "Priya", ("You", "Priya"), 80),
    Conversation("kapoor", "whatsapp_web", "Mr. Kapoor", ("You", "Mr. Kapoor"), 10),
    Conversation("flat-4b", "whatsapp_web", "Flat 4B", ("You", "Rohan", "Meera"), 300),
    Conversation("family", "whatsapp_web", "Family", ("You", "Mom", "Dad", "Anu"), 250),
    Conversation("college", "whatsapp_web", "College gang", ("You", "Kabir", "Zoya"), 300),
)
BY_KEY = {c.key: c for c in CONVERSATIONS}

# Threads that need noise on Nov 25 or 26, before "now", so they fall inside
# the baseline's 48 hour window.
RECENT_NOISE_KEYS = ("sam", "flat-4b", "family", "college")
RECENT_DAYS = (date(2026, 11, 25), date(2026, 11, 26))


def days() -> list[date]:
    """Every day from Nov 2 to Nov 26, inclusive."""
    return [FIRST_DAY + timedelta(days=i) for i in range((LAST_DAY - FIRST_DAY).days + 1)]


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write via a temp file and rename, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def missing_slots(noise: dict[str, Any]) -> list[tuple[str, str]]:
    """(conversation, day) slots that a complete noise.json must have but this one lacks."""
    from corpus.noise import day_plan

    have = noise.get("conversations", {})
    return [
        (conv.key, day)
        for conv in CONVERSATIONS
        for day in day_plan(conv)
        if day not in have.get(conv.key, {})
    ]


def merge(planted: dict[str, Any], noise: dict[str, Any]) -> dict[str, Any]:
    """Merge planted and noise into one scenario in the loader's format."""
    keys = [c["key"] for c in planted["conversations"]]
    if keys != [c.key for c in CONVERSATIONS]:
        raise ValueError(f"planted.json conversations are not the scenario C table: {keys}")

    per_conv: dict[str, list[tuple[datetime, int, int, dict[str, Any]]]] = {k: [] for k in keys}
    for i, msg in enumerate(planted["messages"]):
        per_conv[msg["thread"]].append((repo.parse_time(msg["sent_at"]), 0, i, dict(msg)))

    for key, slots in noise.get("conversations", {}).items():
        if key not in BY_KEY:
            raise ValueError(f"noise.json has unknown conversation {key!r}")
        conv = BY_KEY[key]
        i = 0
        for day in sorted(slots):
            for item in slots[day]:
                sent_at = f"{day}T{item['time']}:00+05:30"
                msg = {
                    "label": None,
                    "app": conv.app,
                    "thread": key,
                    "sender": item["sender"],
                    "sent_at": sent_at,
                    "date_source": DATE_SOURCE[conv.app],
                    "body": item["body"],
                }
                per_conv[key].append((repo.parse_time(sent_at), 1, i, msg))
                i += 1

    messages: list[dict[str, Any]] = []
    n = 0
    for key in keys:
        # Time order; at the same minute a planted message comes before noise.
        for _, _, _, msg in sorted(per_conv[key], key=lambda t: t[:3]):
            if msg["label"] is None:
                n += 1
                msg["label"] = f"n{n:04d}"
            messages.append(msg)

    return {
        "scenario": SCENARIO,
        "description": DESCRIPTION,
        "now": planted["now"],
        "conversations": planted["conversations"],
        "messages": messages,
    }


def cmd_build(args: argparse.Namespace) -> int:
    planted = read_json(args.planted)
    if not Path(args.noise).exists():
        print(f"noise file not found: {args.noise}; run python -m corpus.noise first")
        return 1
    noise = read_json(args.noise)
    missing = missing_slots(noise)
    if missing and not args.allow_partial:
        print(f"{args.noise} is missing {len(missing)} slots, for example {missing[:5]}")
        print("rerun python -m corpus.noise to fill them, or pass --allow-partial")
        return 1
    scenario = merge(planted, noise)
    write_json(args.out, scenario)
    noise_count = sum(1 for m in scenario["messages"] if m["label"].startswith("n"))
    print(f"wrote {len(scenario['messages'])} messages ({noise_count} noise) to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scenario C corpus tools")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="merge planted.json and noise.json")
    build.add_argument("--planted", default=str(PLANTED_PATH))
    build.add_argument("--noise", default=str(NOISE_PATH))
    build.add_argument("--out", default=str(OUT_PATH))
    build.add_argument(
        "--allow-partial", action="store_true", help="merge even if noise.json lacks slots"
    )
    args = parser.parse_args(argv)
    sys.exit(cmd_build(args))


if __name__ == "__main__":
    main()
