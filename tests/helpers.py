"""Shared test builders. All names and data are made up."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from person_context.config import Config
from person_context.models import Event, Speaker, event_id
from person_context.screen import AXNode

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def make_event(
    person_id: int,
    quote: str,
    *,
    text: str = "",
    occurred_at: datetime | None = None,
    source_app: str = "Mail",
    speaker: Speaker = "them",
) -> Event:
    return Event(
        id=event_id(person_id, source_app, quote),
        person_id=person_id,
        occurred_at=occurred_at,
        source_app=source_app,
        source_thread="thread",
        speaker=speaker,
        text=text or quote,
        source_quote=quote,
        ingested_at=T0,
    )


def make_config(db_path: Path, **overrides: object) -> Config:
    values: dict[str, Any] = {
        "gemini_api_key": "test-key",
        "minimi_command": ("minimi-mcp",),
        "minimi_env": {},
        "db_path": db_path,
        "model_compile": "compile-model",
        "model_answer": "answer-model",
        "gap_hours": 6.0,
        "cooldown_hours": 2.0,
        "compile_hours": 6.0,
        "mcp_weekly_budget": 0,
    }
    values.update(overrides)
    return Config(**values)


LRM = "\N{LEFT-TO-RIGHT MARK}"
ENCRYPTION_NOTICE = (
    "Messages and calls are end-to-end encrypted. No one outside of this chat, "
    "not even WhatsApp, can read or listen to them."
)


def whatsapp_tree(name: str, descriptions: list[str], *, composer: str = "") -> AXNode:
    """A native WhatsApp window shaped like the facts in SPEC section 11."""
    cells = [
        AXNode(role="AXCell", identifier="WAMessageBubbleTableViewCell", description=d)
        for d in descriptions
    ]
    return AXNode(
        role="AXWindow",
        children=(
            AXNode(
                role="AXGroup",
                children=(
                    AXNode(
                        role="AXButton",
                        identifier="NavigationBar_HeaderViewButton",
                        title=f"{LRM}{name}",
                    ),
                ),
            ),
            AXNode(
                role="AXTable",
                identifier="ChatMessagesTableView",
                description=f"Messages in chat with {name}",
                children=tuple(AXNode(role="AXRow", children=(cell,)) for cell in cells),
            ),
            AXNode(role="AXTextArea", identifier="ChatBar_ComposerTextView", value=composer),
        ),
    )
