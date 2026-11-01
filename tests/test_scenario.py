"""End to end with fakes: email context about a customer surfaces on their WhatsApp chat."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fakes import FakeLLM, FakeMinimi, FakeMinimiFactory
from helpers import ENCRYPTION_NOTICE, LRM, T0, at, make_config, whatsapp_tree

from person_context.identity import track
from person_context.prompts import ASK_SYSTEM, CARD_SYSTEM, COMPILE_SYSTEM
from person_context.screen import parse_screen
from person_context.service import NO_CONTEXT, Service, describe_source
from person_context.store import Store

ORDER_QUOTE = "Bhavya: Please book 40 chairs for the Pune office, delivery by 14 Oct."
CHANGE_QUOTE = "Bhavya: Change of plan, we need 55 chairs now. Same delivery date."
INVOICE_QUOTE = "Bhavya: Send the invoice to accounts@example.com, not to me."
UNRELATED = "Karan: the landlord is raising my rent, please keep this between us"

MINIMI_PAGE = f"""Found 4 results for Bhavya

---
### Mail, 2 Oct 2026, 11:20
Thread: Chair order
{ORDER_QUOTE}

---
### Mail, 9 Oct 2026, 16:45
Thread: Re: Chair order
{CHANGE_QUOTE}
{INVOICE_QUOTE}

---
### WhatsApp, 9 Oct 2026
Chat: Karan
{UNRELATED}
"""

EXTRACTION = {
    "events": [
        {
            "text": "Bhavya ordered 40 chairs for the Pune office, delivery by 14 Oct.",
            "source_quote": "Please book 40 chairs for the Pune office, delivery by 14 Oct.",
            "occurred_at": "2026-10-02T11:20:00Z",
            "source_app": "Mail",
            "source_thread": "Chair order",
            "speaker": "them",
        },
        {
            "text": "Bhavya changed the order to 55 chairs; delivery date unchanged.",
            "source_quote": "Change of plan, we need 55 chairs now. Same delivery date.",
            "occurred_at": "2026-10-09T16:45:00Z",
            "source_app": "Mail",
            "source_thread": "Re: Chair order",
            "speaker": "them",
        },
        {
            "text": "Bhavya wants invoices sent to accounts@example.com.",
            "source_quote": "Send the invoice to accounts@example.com, not to me.",
            "occurred_at": "2026-10-09T16:45:00Z",
            "source_app": "Mail",
            "source_thread": "Re: Chair order",
            "speaker": "them",
        },
    ],
    "alias_proposals": [],
}

VISIBLE_OUTGOING = (
    f"{LRM}Your message, Hi Bhavya, the 40 chairs are on track, 10:05 AM, Sent to Bhavya, Read"
)
VISIBLE_INCOMING = f"{LRM}Bhavya, any update?, 10:07 AM"


def events_in(prompt: str) -> list[dict[str, Any]]:
    match = re.search(r"<events>\n(.*)\n</events>", prompt, re.DOTALL)
    assert match is not None
    events: list[dict[str, Any]] = json.loads(match.group(1))
    return events


def ref_for(prompt: str, fragment: str) -> str:
    return next(e["event_id"] for e in events_in(prompt) if fragment in e["source_quote"])


def model(model_name: str, system: str, prompt: str) -> Any:
    """Stands in for Gemini, answering from what the prompt actually contains."""
    if system == COMPILE_SYSTEM:
        return EXTRACTION
    if system == CARD_SYSTEM:
        return {
            "show": True,
            "lines": [
                {
                    "text": "Order changed from 40 to 55 chairs on 9 Oct; delivery still 14 Oct.",
                    "event_ids": [ref_for(prompt, "55 chairs"), ref_for(prompt, "40 chairs")],
                    "quote": "we need 55 chairs now",
                },
                {
                    "text": "Invoices go to accounts@example.com.",
                    "event_ids": [ref_for(prompt, "invoice")],
                    "quote": "Send the invoice to accounts@example.com",
                },
                {
                    "text": "The 40 chairs are on track.",
                    "event_ids": [ref_for(prompt, "40 chairs")],
                    "quote": "40 chairs",
                },
                {
                    "text": "Bhavya asked for a discount.",
                    "event_ids": [ref_for(prompt, "40 chairs")],
                    "quote": "please give us a 10% discount",
                },
            ],
        }
    assert system == ASK_SYSTEM
    return {
        "answer": "55 chairs. This changed on 9 Oct: the original order on 2 Oct was 40 chairs.",
        "citations": [
            {"event_id": ref_for(prompt, "55 chairs"), "quote": "we need 55 chairs now"},
            {"event_id": ref_for(prompt, "40 chairs"), "quote": "Please book 40 chairs"},
        ],
    }


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> Any:
        return self.now


def test_customer_scenario(tmp_path: Path) -> None:
    cfg = make_config(tmp_path / "context.db", mcp_weekly_budget=20)
    store = Store.open(cfg.db_path)
    track(store, "Bhavya", T0)
    llm = FakeLLM(handler=model)
    minimi = FakeMinimi({"Bhavya": [MINIMI_PAGE]})
    factory = FakeMinimiFactory(minimi, weekly_limit=cfg.mcp_weekly_budget)
    clock = Clock()
    service = Service(cfg, lambda: Store.open(cfg.db_path), llm, factory, clock=clock)

    # The scheduled compile stores the email context.
    [result] = service.compile() or []
    assert result.outcome == "stored" and result.events_added == 3

    # Later the user opens Bhavya's WhatsApp chat, which shows only a vague message.
    clock.now = at(26)
    screen = parse_screen(
        whatsapp_tree(
            "Bhavya", [ENCRYPTION_NOTICE, VISIBLE_OUTGOING, VISIBLE_INCOMING], composer="typing"
        )
    )
    assert screen is not None and screen.person_name == "Bhavya"
    card = service.on_screen(screen)

    assert card is not None
    assert [line.text for line in card.lines] == [
        "Order changed from 40 to 55 chairs on 9 Oct; delivery still 14 Oct.",
        "Invoices go to accounts@example.com.",
    ]
    assert describe_source(card.lines[0].source) == ("“we need 55 chairs now” · Mail · 09 Oct 2026")
    assert store.event_count(1) == 3  # freshness re-read the same memories without duplicates

    card_prompt = next(c["prompt"] for c in llm.calls if c["system"] == CARD_SYSTEM)
    assert "any update?" in card_prompt
    assert "typing" not in card_prompt
    assert "landlord" not in llm.prompts() and "Karan" not in llm.prompts()

    # Staying on the chat, then opening an unrelated chat, fires nothing.
    calls = len(llm.calls)
    clock.now = at(26.1)
    assert service.on_screen(screen) is None
    unrelated = parse_screen(whatsapp_tree("Karan", [f"{LRM}Karan, hey, 9:00 AM"]))
    assert unrelated is not None
    assert service.on_screen(unrelated) is None
    assert len(llm.calls) == calls

    # Ask answers with citations and flags the changed number.
    answer = service.ask("How many chairs does she need?", screen)
    assert answer.status == "answered"
    assert "55 chairs" in answer.answer and "40 chairs" in answer.answer
    assert [describe_source(s) for s in answer.sources] == [
        "“we need 55 chairs now” · Mail · 09 Oct 2026",
        "“Please book 40 chairs” · Mail · 02 Oct 2026",
    ]
    ask_prompt = next(c["prompt"] for c in llm.calls if c["system"] == ASK_SYSTEM)
    assert ask_prompt.index("40 chairs") < ask_prompt.index("55 chairs")
    assert "flag it in the answer" in ASK_SYSTEM

    # An answer whose citations do not survive grounding is not shown.
    llm.handler = lambda m, s, p: (
        model(m, s, p)
        if s != ASK_SYSTEM
        else {"answer": "60 chairs.", "citations": [{"event_id": "0" * 12, "quote": "60 chairs"}]}
    )
    assert service.ask("How many chairs?", screen).answer == NO_CONTEXT
