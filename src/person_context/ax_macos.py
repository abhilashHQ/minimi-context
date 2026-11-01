"""Reads native WhatsApp's accessibility tree into AXNode. macOS only."""

from __future__ import annotations

from typing import Any

from AppKit import NSWorkspace
from ApplicationServices import (
    AXIsProcessTrustedWithOptions,
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    AXUIElementSetMessagingTimeout,
)

from .screen import CHAT_LIST_ID, COMPOSER_ID, AXNode

WHATSAPP_BUNDLE_ID = "net.whatsapp.WhatsApp"
_AX_SUCCESS = 0
_MESSAGING_TIMEOUT_SECONDS = 1.0
# Bounds keep one read fast even when a long chat history is loaded.
_MAX_NODES = 5000
_MAX_DEPTH = 40


def ensure_trusted() -> bool:
    """True if this process may read other apps; asks the user once if not."""
    return bool(AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True}))


def frontmost_whatsapp_pid() -> int | None:
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None or app.bundleIdentifier() != WHATSAPP_BUNDLE_ID:
        return None
    return int(app.processIdentifier())


def read_window(pid: int) -> AXNode | None:
    """The focused WhatsApp window as an AXNode tree, or None if there is none."""
    app = AXUIElementCreateApplication(pid)
    AXUIElementSetMessagingTimeout(app, _MESSAGING_TIMEOUT_SECONDS)
    window = _attribute(app, "AXFocusedWindow")
    # While the session is locked this returns the application element, whose
    # children include itself; reading it would exhaust the node budget every poll.
    if window is None or _text(window, "AXRole") != "AXWindow":
        return None
    remaining = [_MAX_NODES]
    return _node(window, 0, remaining)


def _attribute(element: Any, name: str) -> Any:
    error, value = AXUIElementCopyAttributeValue(element, name, None)
    return value if error == _AX_SUCCESS else None


def _text(element: Any, name: str) -> str | None:
    value = _attribute(element, name)
    return str(value) if isinstance(value, str) else None


def _node(element: Any, depth: int, remaining: list[int]) -> AXNode:
    remaining[0] -= 1
    role = _text(element, "AXRole")
    identifier = _text(element, "AXIdentifier")
    if identifier in (COMPOSER_ID, CHAT_LIST_ID):
        # Never read the composer (the unsent draft) or the chat list (other chats'
        # previews).
        return AXNode(role=role, identifier=identifier)
    children: list[AXNode] = []
    if depth < _MAX_DEPTH:
        for child in _attribute(element, "AXChildren") or ():
            if remaining[0] <= 0:
                break
            children.append(_node(child, depth + 1, remaining))
    return AXNode(
        role=role,
        identifier=identifier,
        title=_text(element, "AXTitle"),
        value=_text(element, "AXValue"),
        description=_text(element, "AXDescription"),
        children=tuple(children),
    )
