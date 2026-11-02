"""Parse the independent copy of the scenario C spec tables in fixtures/."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

SPEC = Path(__file__).with_name("fixtures") / "scenario_c_spec.md"


def _rows() -> list[list[str]]:
    rows = []
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith("|---"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def conversation_rows() -> list[dict[str, str]]:
    return [
        dict(zip(("key", "app", "title", "planted", "noise"), r, strict=True))
        for r in _rows()
        if len(r) == 5 and r[1] in ("Gmail", "WhatsApp")
    ]


def planted_rows() -> list[dict[str, str]]:
    return [
        dict(zip(("id", "thread", "sent", "from", "text"), r, strict=True))
        for r in _rows()
        if len(r) == 5 and r[0][:1] == "c" and r[0][1:].isdigit()
    ]


def question_rows() -> list[dict[str, str]]:
    return [
        dict(zip(("id", "question", "required"), r, strict=True))
        for r in _rows()
        if len(r) == 3 and r[0][:1] == "C" and r[0][1:].isdigit()
    ]


def parse_sent(sent: str) -> datetime:
    """'Wed Nov 18, 10:15 AM' in 2026. Checks the weekday name against the date."""
    weekday, rest = sent.split(" ", 1)
    dt = datetime.strptime(f"{rest} 2026", "%b %d, %I:%M %p %Y")
    assert dt.strftime("%a") == weekday, f"{sent}: 2026 date falls on {dt:%a}"
    return dt
