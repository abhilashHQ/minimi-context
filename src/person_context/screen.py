"""Pure parser from WhatsApp's accessibility tree to a ScreenContext."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .models import Speaker, normalize

HEADER_ID = "NavigationBar_HeaderViewButton"
MESSAGE_LIST_ID = "ChatMessagesTableView"
MESSAGE_ID = "WAMessageBubbleTableViewCell"
COMPOSER_ID = "ChatBar_ComposerTextView"
CHAT_LIST_ID = "ChatListView_TableView"

_SELF_PREFIX = "Your message,"
_ENCRYPTION_NOTICE = "end-to-end encrypted"
_DIRECTION_MARKS = "\N{LEFT-TO-RIGHT MARK}\N{RIGHT-TO-LEFT MARK}"


@dataclass(frozen=True)
class AXNode:
    role: str | None = None
    identifier: str | None = None
    title: str | None = None
    value: str | None = None
    description: str | None = None
    children: Sequence[AXNode] = ()


@dataclass(frozen=True)
class ScreenMessage:
    speaker: Speaker
    text: str


@dataclass(frozen=True)
class ScreenContext:
    person_name: str
    messages: tuple[ScreenMessage, ...]


def parse_screen(root: AXNode) -> ScreenContext | None:
    """The open chat's name and visible messages, or None if no chat is open."""
    header = next(
        (n for n in _walk(root) if n.role == "AXButton" and n.identifier == HEADER_ID), None
    )
    message_list = next((n for n in _walk(root) if n.identifier == MESSAGE_LIST_ID), None)
    if header is None or message_list is None:
        return None
    name = _clean(header.title) or _clean(header.description)
    if not name:
        return None
    messages = tuple(
        message
        for node in _walk(message_list)
        if node.identifier == MESSAGE_ID
        for message in (_message(node),)
        if message is not None
    )
    return ScreenContext(person_name=name, messages=messages)


def _walk(node: AXNode) -> Iterator[AXNode]:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.identifier == COMPOSER_ID:
            continue
        yield current
        stack.extend(reversed(current.children))


def _clean(text: str | None) -> str:
    return (text or "").strip().strip(_DIRECTION_MARKS).strip()


def _message(cell: AXNode) -> ScreenMessage | None:
    if cell.description and _clean(cell.description):
        text = cell.description
        if _ENCRYPTION_NOTICE in normalize(text):
            return None
        speaker: Speaker = "self" if _clean(text).startswith(_SELF_PREFIX) else "them"
        return ScreenMessage(speaker, text)
    # Unknown shape: keep whatever text the cell or its children carry.
    text = " ".join(
        part
        for node in _walk(cell)
        for part in (node.description, node.value, node.title)
        if part and _clean(part)
    )
    if not text or _ENCRYPTION_NOTICE in normalize(text):
        return None
    return ScreenMessage("unknown", text)
