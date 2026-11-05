"""Small shared helpers: names, dates, weekday checks, quote matching, embeddings."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher

import numpy as np

HONORIFICS = {"mr.", "mrs.", "ms.", "dr.", "mr", "mrs", "ms", "dr"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_RE = re.compile(r"\b(mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(day|nesday|sday|urday|rsday)?\b", re.I)
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[\s<>]+", name.lower()) if t and t not in HONORIFICS and "@" not in t]


def same_person(a: str, b: str) -> bool:
    """First-name prefix match: "Sam" matches "Samuel Rao", "Priya" matches "Priya Nair"."""
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb or "you" in (ta[0], tb[0]):
        return False
    return ta[0].startswith(tb[0]) or tb[0].startswith(ta[0])


def overlap(people_a, people_b) -> bool:
    return any(same_person(x, y) for x in people_a or [] for y in people_b or [])


def is_you(name: str) -> bool:
    return name.strip().lower() in ("you", "me")


def dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def iso(d: datetime) -> str:
    return d.isoformat(timespec="seconds")


def fmt_day(ts: str) -> str:
    return dt(ts).strftime("%a %Y-%m-%d %H:%M")


def human_time(ts: str, with_time: bool = True) -> str:
    d = dt(ts)
    day = f"{d:%a} {d:%b} {d.day}"
    if not with_time or (d.hour, d.minute) == (23, 59):
        return day
    h = d.hour % 12 or 12
    mins = f":{d.minute:02d}" if d.minute else ""
    return f"{day}, {h}{mins} {'AM' if d.hour < 12 else 'PM'}"


def parse_event_at(value: str | None, tz_from: str) -> str | None:
    """Normalize the model's event_at ("YYYY-MM-DD" or "YYYY-MM-DDTHH:MM") to ISO with the
    source's UTC offset. Returns None when it doesn't parse."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip().replace(" ", "T")
    try:
        d = datetime.fromisoformat(value)
    except ValueError:
        return None
    if len(value) == 10:
        d = d.replace(hour=23, minute=59)
    tzinfo = dt(tz_from).tzinfo
    return iso(d.replace(tzinfo=tzinfo) if d.tzinfo is None else d.astimezone(tzinfo))


def weekday_in(text: str) -> int | None:
    """The single weekday named in a text, or None when there is none or more than one."""
    found = {m.group(1).lower()[:3] for m in WEEKDAY_RE.finditer(text)}
    if len(found) != 1:
        return None
    return [w[:3] for w in WEEKDAYS].index(found.pop())


def snap_weekday(event_at: str, quote: str, said_at: str) -> str | None:
    """If the quote names one weekday and event_at falls on another day, return event_at moved to
    the first such weekday after said_at (same time of day). Otherwise None."""
    wd = weekday_in(quote)
    if wd is None:
        return None
    ev = dt(event_at)
    if ev.weekday() == wd:
        return None
    said = dt(said_at).date()
    step = (wd - said.weekday()) % 7 or 7
    if re.search(r"\bnext\s+week\b", quote, re.I):
        step += 7
    target: date = said + timedelta(days=step)
    return iso(ev.replace(year=target.year, month=target.month, day=target.day))


def rewrite_dates(text: str, old: str, new: str) -> str:
    """Replace mentions of the old date ("Wed Nov 26", "November 26", "Nov 26") with the new one."""
    o, n = dt(old), dt(new)
    if o.date() == n.date():
        return text
    o_wd, n_wd = o.strftime("%a"), n.strftime("%a")
    o_wdl, n_wdl = o.strftime("%A"), n.strftime("%A")
    o_mon, n_mon = o.strftime("%b"), n.strftime("%b")
    o_monl, n_monl = o.strftime("%B"), n.strftime("%B")
    out = re.sub(rf"\b{o_monl}\s+{o.day}\b", f"{n_monl} {n.day}", text)
    out = re.sub(rf"\b{o_mon}\s+{o.day}\b", f"{n_mon} {n.day}", out)
    out = re.sub(rf"\b{o.day}\s+{o_mon}", f"{n.day} {n_mon}", out)
    out = re.sub(rf"\b{o_wdl}\b", n_wdl, out)
    out = re.sub(rf"\b{o_wd}\b", n_wd, out)
    return out


def quote_ok(quote: str, body: str) -> bool:
    q = " ".join((quote or "").lower().split())
    b = " ".join(body.lower().split())
    if not q:
        return False
    if q in b:
        return True
    m = SequenceMatcher(None, q, b, autojunk=False).find_longest_match(0, len(q), 0, len(b))
    return m.size >= 0.9 * len(q)


class Embedder:
    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def many(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def one(self, text: str) -> np.ndarray:
        return self.many([text])[0]


class LazyEmbedder:
    """Loads the model on first use, so a server can answer initialize before torch is imported."""

    def __init__(self) -> None:
        self._inner: Embedder | None = None

    def _get(self) -> Embedder:
        if self._inner is None:
            self._inner = Embedder()
        return self._inner

    def many(self, texts: list[str]) -> np.ndarray:
        return self._get().many(texts)

    def one(self, text: str) -> np.ndarray:
        return self._get().one(text)
