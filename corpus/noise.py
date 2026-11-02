"""Generate the scenario C noise once, then freeze it.

    python -m corpus.noise --out corpus/scenario_c/noise.json

Works one conversation and one day at a time. Each call asks the LLM for that
day's messages; every message is checked with the same rule functions as
`corpus.check`, and messages that fail are dropped and asked for again until
the day's target is met. The file is written after every slot, so a crash
loses at most one slot: rerunning skips every (conversation, day) already in
the file. Existing slots are never replaced; `--force` starts from an empty
file instead.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from corpus.check import BANNED_TERMS, YOU_BANNED_TERMS, noise_rule_violations
from corpus.scenario_c import (
    CONVERSATIONS,
    NOISE_PATH,
    PLANTED_PATH,
    RECENT_DAYS,
    RECENT_NOISE_KEYS,
    SCENARIO,
    Conversation,
    days,
    read_json,
    write_json,
)
from sme.llm.client import LLMClient
from sme.store import repo

# Messages may not land this close to a planted message in the same
# conversation, so planted questions and their replies stay adjacent.
QUIET_MINUTES = 15
# The last day ends at "now", 20:00; noise there must be strictly earlier.
LAST_DAY_END = "19:59"
MAX_ATTEMPTS = 8
CONTEXT_MESSAGES = 8
TEMPERATURE = 0.9


@dataclass(frozen=True)
class Persona:
    about: str
    topics: tuple[str, ...]
    # Plausible hours, HH:MM 24 hour, inclusive.
    start: str
    end: str
    style: str = "WhatsApp chat: short, casual, one or two sentences."


PERSONAS: dict[str, Persona] = {
    "standup": Persona(
        "Dev Mehta, your manager, emails the team's daily standup notes.",
        ("standup status lines", "work gossip"),
        "09:30",
        "09:30",
        "Work email: three to five plain status lines, one per teammate or project.",
    ),
    "newsletter": Persona(
        "The Weekly Byte is a tech newsletter that lands in your inbox every weekday.",
        ("newsletter tech stories",),
        "07:00",
        "07:00",
        "Newsletter email: a short issue with three or four one-line tech stories.",
    ),
    "sam": Persona(
        "Sam is your close friend. Easy banter between the two of you.",
        ("football", "films", "memes described in words", "work gossip"),
        "08:00",
        "23:45",
    ),
    "priya": Persona(
        "Priya is a friend from work.",
        ("work gossip", "films", "memes described in words"),
        "09:00",
        "22:00",
    ),
    "kapoor": Persona(
        "Mr. Kapoor owns the flat you rent. He sends rent receipts and building notices"
        " (water supply timings, lift maintenance, parking stickers, stairwell painting)."
        " You reply briefly to acknowledge.",
        ("rent receipts", "building notices"),
        "09:00",
        "19:00",
    ),
    "flat-4b": Persona(
        "Rohan and Meera are your flatmates in Flat 4B. The three of you share a group chat.",
        ("household chores already done", "memes described in words", "football", "films"),
        "07:30",
        "23:45",
    ),
    "family": Persona(
        "Your family group chat with Mom, Dad and Anu.",
        ("family news", "memes described in words", "films"),
        "07:00",
        "22:30",
    ),
    "college": Persona(
        "Kabir and Zoya are your friends from college. The three of you share a group chat.",
        ("college memories", "football", "films", "memes described in words"),
        "08:00",
        "23:59",
    ),
}

NOISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sender": {"type": "string"},
                    "time": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["sender", "time", "body"],
            },
        }
    },
    "required": ["messages"],
}

PROMPT = """You write background chatter for a fictional test corpus. These messages are noise:
they must carry no plans, invitations, requests, favours, promises, purchases, money, travel,
food or meals, and no days of the week or dates.

Conversation: {title} ({app})
About: {about}
Topics: {topics}
Style: {style}

Write exactly {n} new messages sent on {day}, {kind_of_day}.
Senders: only {senders}.
Times: 24 hour HH:MM between {start} and {end}.{avoid}
Never use these words, in any form: {banned}
Messages from You also never use: {you_banned}
Things already done or already happened are fine. Nobody asks anybody to do anything.
{recent}{rejected}
Reply as JSON: {{"messages": [{{"sender": "...", "time": "HH:MM", "body": "..."}}]}}
"""

APP_NAMES = {"whatsapp_web": "WhatsApp", "gmail": "Gmail"}

Slots = dict[str, dict[str, list[dict[str, str]]]]


class NoiseGenerationError(RuntimeError):
    pass


def day_plan(conv: Conversation) -> dict[str, int]:
    """How many noise messages each day of a conversation gets. Sums to its target.

    Fixed-schedule conversations get one message each weekday. Others spread
    the target over Nov 2 to 26 with seeded random weights, more at weekends,
    less on the short last day. Deterministic for a given conversation.
    """
    if conv.noise_target == 0:
        return {}
    all_days = days()
    if conv.weekday_time:
        return {d.isoformat(): 1 for d in all_days if d.weekday() < 5}

    rng = random.Random(f"scenario-c-noise:{conv.key}")
    weights = []
    for d in all_days:
        w = rng.uniform(0.6, 1.4)
        if d.weekday() >= 5:
            w *= 1.3
        if d == all_days[-1]:
            w *= 0.6
        weights.append(w)
    total = sum(weights)
    raw = [conv.noise_target * w / total for w in weights]
    counts = [int(x) for x in raw]
    by_remainder = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in by_remainder[: conv.noise_target - sum(counts)]:
        counts[i] += 1

    if conv.key in RECENT_NOISE_KEYS:
        recent = [all_days.index(d) for d in RECENT_DAYS]
        if not any(counts[i] for i in recent):
            counts[counts.index(max(counts))] -= 1
            counts[recent[-1]] += 1
    return {d.isoformat(): n for d, n in zip(all_days, counts, strict=True)}


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def quiet_windows(planted: dict[str, Any]) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """(conversation, day) -> minute ranges around planted messages where noise may not go.

    Ranges are clipped to the planted message's own day; no planted message is
    within QUIET_MINUTES of midnight.
    """
    out: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for msg in planted["messages"]:
        sent = repo.parse_time(msg["sent_at"])
        minute = sent.hour * 60 + sent.minute
        span = (max(0, minute - QUIET_MINUTES), min(24 * 60 - 1, minute + QUIET_MINUTES))
        out.setdefault((msg["thread"], sent.date().isoformat()), []).append(span)
    return out


def slot_window(conv: Conversation, day: str) -> tuple[str, str]:
    if conv.weekday_time:
        return conv.weekday_time, conv.weekday_time
    persona = PERSONAS[conv.key]
    end = persona.end
    if date.fromisoformat(day) == days()[-1] and _minutes(end) > _minutes(LAST_DAY_END):
        end = LAST_DAY_END
    return persona.start, end


def message_problems(
    conv: Conversation,
    day: str,
    item: Any,
    quiet: list[tuple[int, int]],
    used_bodies: set[str],
) -> list[str]:
    """Why one generated message cannot be used. Empty means keep it."""
    if not isinstance(item, dict):
        return ["not an object"]
    sender = str(item.get("sender", "")).strip()
    time = str(item.get("time", "")).strip()
    body = str(item.get("body", "")).strip()
    problems = []
    if not body:
        problems.append("empty body")
    try:
        datetime.strptime(time, "%H:%M")
        minutes = _minutes(time)
    except ValueError:
        problems.append(f"time {time!r} is not HH:MM")
        minutes = None
    if minutes is not None:
        start, end = slot_window(conv, day)
        if not _minutes(start) <= minutes <= _minutes(end):
            problems.append(f"time {time} is outside {start} to {end}")
        elif any(lo <= minutes <= hi for lo, hi in quiet):
            problems.append(f"time {time} is too close to another message")
    problems += noise_rule_violations(sender, body, conv.members)
    if repo.normalize_body(body) in used_bodies:
        problems.append("repeats an earlier message")
    return problems


def build_prompt(
    conv: Conversation,
    day: str,
    n: int,
    senders: tuple[str, ...],
    quiet: list[tuple[int, int]],
    recent: list[dict[str, str]],
    rejected: list[tuple[dict[str, Any], list[str]]],
) -> str:
    persona = PERSONAS[conv.key]
    start, end = slot_window(conv, day)
    avoid = ""
    if quiet:
        spans = ", ".join(f"{_hhmm(lo)} to {_hhmm(hi)}" for lo, hi in quiet)
        avoid = f" Do not use these times: {spans}."
    recent_text = ""
    if recent:
        lines = "\n".join(f"- {m['day']} {m['time']} {m['sender']}: {m['body']}" for m in recent)
        recent_text = f"Recent messages, for continuity only, do not repeat them:\n{lines}\n"
    rejected_text = ""
    if rejected:
        lines = "\n".join(f"- {item!r}: {'; '.join(reasons)}" for item, reasons in rejected[-10:])
        rejected_text = f"These were rejected; write different ones:\n{lines}\n"
    return PROMPT.format(
        title=conv.title,
        app=APP_NAMES[conv.app],
        about=persona.about,
        topics=", ".join(persona.topics),
        style=persona.style,
        n=n,
        kind_of_day="a weekend day" if date.fromisoformat(day).weekday() >= 5 else "a weekday",
        day=day,
        senders=", ".join(senders),
        start=start,
        end=end,
        avoid=avoid,
        banned=", ".join(BANNED_TERMS),
        you_banned=", ".join(YOU_BANNED_TERMS),
        recent=recent_text,
        rejected=rejected_text,
    )


def generate_slot(
    llm: LLMClient,
    conv: Conversation,
    day: str,
    n: int,
    quiet: list[tuple[int, int]],
    recent: list[dict[str, str]],
    used_bodies: set[str],
    max_attempts: int = MAX_ATTEMPTS,
) -> list[dict[str, str]]:
    """Ask until `n` messages pass every rule. Returns them sorted by time."""
    # Fixed-schedule threads have one sender at one time; the model writes the body.
    senders = conv.members[1:2] if conv.weekday_time else conv.members
    accepted: list[dict[str, str]] = []
    rejected: list[tuple[dict[str, Any], list[str]]] = []
    seen = set(used_bodies)
    for _ in range(max_attempts):
        need = n - len(accepted)
        context = (recent + [{"day": day, **m} for m in accepted])[-CONTEXT_MESSAGES:]
        reply = llm.generate_json(
            build_prompt(conv, day, need, senders, quiet, context, rejected), NOISE_SCHEMA
        )
        items = reply.get("messages", []) if isinstance(reply, dict) else []
        for item in items:
            if len(accepted) == n:
                break
            if conv.weekday_time and isinstance(item, dict):
                item = {**item, "sender": senders[0], "time": conv.weekday_time}
            problems = message_problems(conv, day, item, quiet, seen)
            if problems:
                rejected.append((item, problems))
                continue
            msg = {
                "sender": str(item["sender"]).strip(),
                "time": str(item["time"]).strip(),
                "body": str(item["body"]).strip(),
            }
            accepted.append(msg)
            seen.add(repo.normalize_body(msg["body"]))
        if len(accepted) == n:
            return sorted(accepted, key=lambda m: m["time"])
    raise NoiseGenerationError(
        f"{conv.key} {day}: only {len(accepted)} of {n} messages passed after"
        f" {max_attempts} attempts; last rejections: {rejected[-3:]!r}"
    )


def _ordered(noise: dict[str, Any]) -> dict[str, Any]:
    """Copy with days sorted, for stable diffs."""
    convs = {k: dict(sorted(v.items())) for k, v in noise["conversations"].items()}
    return {**noise, "conversations": convs}


def empty_noise(model: str | None) -> dict[str, Any]:
    return {"scenario": SCENARIO, "model": model, "conversations": {}}


def generate(
    llm: LLMClient,
    out_path: str | Path,
    planted: dict[str, Any],
    force: bool = False,
    only: set[str] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fill every missing (conversation, day) slot of `out_path`, saving after each one."""
    log = log or (lambda line: print(line, flush=True))
    out_path = Path(out_path)
    model = getattr(llm, "model", None)
    if out_path.exists() and not force:
        noise = read_json(out_path)
        if noise.get("scenario") != SCENARIO:
            raise ValueError(f"{out_path} is not a scenario C noise file")
        log(f"resuming {out_path}")
    else:
        noise = empty_noise(model)
    slots: Slots = noise.setdefault("conversations", {})
    quiet = quiet_windows(planted)

    for conv in CONVERSATIONS:
        plan = day_plan(conv)
        if not plan or (only and conv.key not in only):
            continue
        conv_slots = slots.setdefault(conv.key, {})
        done = sum(1 for d in plan if d in conv_slots)
        log(f"[{conv.key}] {done}/{len(plan)} days already done, target {conv.noise_target}")
        for day, n in plan.items():
            if day in conv_slots:
                continue
            if n:
                recent = [
                    {"day": d, **m} for d in sorted(conv_slots) if d < day for m in conv_slots[d]
                ][-CONTEXT_MESSAGES:]
                used = {repo.normalize_body(m["body"]) for v in conv_slots.values() for m in v}
                conv_slots[day] = generate_slot(
                    llm, conv, day, n, quiet.get((conv.key, day), []), recent, used, max_attempts
                )
            else:
                conv_slots[day] = []
            write_json(out_path, _ordered(noise))
            total = sum(len(v) for v in conv_slots.values())
            log(f"[{conv.key}] {day}: {n} messages ({total}/{conv.noise_target})")
    noise = _ordered(noise)
    write_json(out_path, noise)
    return noise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the scenario C noise")
    parser.add_argument("--out", default=str(NOISE_PATH), help="noise JSON path")
    parser.add_argument("--planted", default=str(PLANTED_PATH), help="planted messages JSON")
    parser.add_argument(
        "--force", action="store_true", help="discard an existing noise file and start over"
    )
    parser.add_argument("--only", help="comma separated conversation keys to fill")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    args = parser.parse_args(argv)

    from sme.llm.client import GeminiClient

    llm = GeminiClient(temperature=TEMPERATURE)
    only = set(args.only.split(",")) if args.only else None
    try:
        noise = generate(
            llm,
            args.out,
            read_json(args.planted),
            force=args.force,
            only=only,
            max_attempts=args.max_attempts,
        )
    except NoiseGenerationError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        print("finished slots are saved; rerun the same command to resume", file=sys.stderr)
        sys.exit(1)
    total = sum(len(v) for c in noise["conversations"].values() for v in c.values())
    print(f"{args.out}: {total} noise messages")


if __name__ == "__main__":
    main()
