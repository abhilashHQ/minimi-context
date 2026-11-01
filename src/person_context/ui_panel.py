"""Card, Ask and alias-confirm UI in one non-activating panel. macOS only, main thread only."""

from __future__ import annotations

from typing import Any, Callable

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSEvent,
    NSFloatingWindowLevel,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSPointInRect,
    NSScreen,
    NSStackView,
    NSTextField,
    NSTimer,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSObject

from .models import Alias
from .service import AskResult, Card, describe_source

WIDTH = 360.0
PADDING = 14.0
SCREEN_MARGIN = 16.0
AUTO_HIDE_SECONDS = 45.0
RECHECK_SECONDS = 5.0
WORKING = "Working"

_VERTICAL = 1  # NSUserInterfaceLayoutOrientationVertical
_HORIZONTAL = 0  # NSUserInterfaceLayoutOrientationHorizontal
_LEADING = 5  # NSLayoutAttributeLeading
_MATERIAL_POPOVER = 6  # NSVisualEffectMaterialPopover
_STATE_ACTIVE = 1  # NSVisualEffectStateActive


class _KeyablePanel(NSPanel):
    # The panel never takes focus on its own; this lets the Ask field accept typing
    # after a click or the hotkey.
    def canBecomeKeyWindow(self) -> bool:
        return True

    def canBecomeMainWindow(self) -> bool:
        return False


class _Actions(NSObject):
    """Target for buttons and the Ask field. `owner` is set right after init."""

    @objc.IBAction
    def ask_(self, sender: Any) -> None:
        self.owner._submit(str(sender.stringValue()))

    @objc.IBAction
    def dismiss_(self, sender: Any) -> None:
        self.owner.hide()

    @objc.IBAction
    def confirmAlias_(self, sender: Any) -> None:
        self.owner._resolve(True)

    @objc.IBAction
    def rejectAlias_(self, sender: Any) -> None:
        self.owner._resolve(False)


class ContextPanel:
    def __init__(
        self,
        on_ask: Callable[[str], None],
        on_resolve_alias: Callable[[Alias, bool], None],
    ) -> None:
        self._on_ask = on_ask
        self._on_resolve_alias = on_resolve_alias
        self._card: Card | None = None
        self._title = ""
        self._answer: list[str] = []
        self._busy = False
        self._timer: Any = None
        self._field: Any = None
        self._actions = _Actions.alloc().init()
        self._actions.owner = self

        panel = _KeyablePanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIDTH, 100),
            NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setFloatingPanel_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)

        background = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, WIDTH, 100))
        background.setMaterial_(_MATERIAL_POPOVER)
        background.setState_(_STATE_ACTIVE)
        background.setWantsLayer_(True)
        background.layer().setCornerRadius_(10.0)
        background.layer().setMasksToBounds_(True)
        panel.setContentView_(background)

        stack = NSStackView.alloc().init()
        stack.setOrientation_(_VERTICAL)
        stack.setAlignment_(_LEADING)
        stack.setSpacing_(8.0)
        stack.setEdgeInsets_((PADDING, PADDING, PADDING, PADDING))
        stack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        background.addSubview_(stack)
        for anchor in ("leadingAnchor", "trailingAnchor", "topAnchor", "bottomAnchor"):
            getattr(stack, anchor)().constraintEqualToAnchor_(
                getattr(background, anchor)()
            ).setActive_(True)
        stack.widthAnchor().constraintEqualToConstant_(WIDTH).setActive_(True)

        self._panel = panel
        self._stack = stack

    # Called by the app on the main thread

    def show_card(self, card: Card) -> None:
        """Show without taking focus from WhatsApp. Never replaces an Ask in use."""
        if self._busy or self._panel.isKeyWindow():
            return
        self._card = card
        self._title = card.person.display_name
        self._answer = []
        self._render()
        self._panel.orderFrontRegardless()
        self._arm_auto_hide()

    def toggle_ask(self, person_name: str | None) -> None:
        if self._panel.isVisible():
            self.hide()
            return
        self._card = None
        self._title = person_name or "Ask"
        self._answer = []
        self._render()
        self._panel.makeKeyAndOrderFront_(None)
        self._panel.makeFirstResponder_(self._field)
        self._arm_auto_hide()

    def show_answer(self, result: AskResult) -> None:
        self._busy = False
        self._answer = [result.answer] + [describe_source(s) for s in result.sources]
        self._render()
        self._arm_auto_hide()

    def hide(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._panel.orderOut_(None)

    # Actions

    def _submit(self, question: str) -> None:
        question = question.strip()
        if not question or self._busy:
            return
        self._busy = True
        self._answer = [WORKING]
        self._render()
        self._on_ask(question)

    def _resolve(self, confirm: bool) -> None:
        card = self._card
        if card is None or card.pending_alias is None:
            return
        self._on_resolve_alias(card.pending_alias, confirm)
        self._card = Card(card.person, card.lines, None)
        self._render()

    # Layout

    def _render(self) -> None:
        for view in list(self._stack.arrangedSubviews()):
            self._stack.removeArrangedSubview_(view)
            view.removeFromSuperview()
        self._add(_label(self._title, NSFont.boldSystemFontOfSize_(13.0), NSColor.labelColor()))
        if self._card is not None:
            for line in self._card.lines:
                self._add(_label(line.text, NSFont.systemFontOfSize_(13.0), NSColor.labelColor()))
                self._add(
                    _label(
                        describe_source(line.source),
                        NSFont.systemFontOfSize_(11.0),
                        NSColor.secondaryLabelColor(),
                    )
                )
            if self._card.pending_alias is not None:
                self._add(self._alias_row(self._card.pending_alias))
        for index, text in enumerate(self._answer):
            font = NSFont.systemFontOfSize_(13.0 if index == 0 else 11.0)
            color = NSColor.labelColor() if index == 0 else NSColor.secondaryLabelColor()
            self._add(_label(text, font, color))
        field = NSTextField.textFieldWithString_("")
        field.setPlaceholderString_("Ask about this person")
        field.setTarget_(self._actions)
        field.setAction_("ask:")
        field.setEnabled_(not self._busy)
        field.widthAnchor().constraintEqualToConstant_(WIDTH - 2 * PADDING).setActive_(True)
        self._field = field
        self._add(field)
        self._add(NSButton.buttonWithTitle_target_action_("Dismiss", self._actions, "dismiss:"))
        self._place()

    def _alias_row(self, alias: Alias) -> Any:
        text = f"Is {alias.value} also {self._title}?"
        row = NSStackView.stackViewWithViews_(
            [
                _label(text, NSFont.systemFontOfSize_(11.0), NSColor.secondaryLabelColor()),
                NSButton.buttonWithTitle_target_action_("Confirm", self._actions, "confirmAlias:"),
                NSButton.buttonWithTitle_target_action_("Reject", self._actions, "rejectAlias:"),
            ]
        )
        row.setOrientation_(_HORIZONTAL)
        return row

    def _add(self, view: Any) -> None:
        self._stack.addArrangedSubview_(view)

    def _place(self) -> None:
        self._stack.layoutSubtreeIfNeeded()
        height = self._stack.fittingSize().height
        visible = NSScreen.mainScreen().visibleFrame()
        x = visible.origin.x + visible.size.width - WIDTH - SCREEN_MARGIN
        y = visible.origin.y + SCREEN_MARGIN
        self._panel.setFrame_display_(NSMakeRect(x, y, WIDTH, height), True)

    # Auto-hide

    def _arm_auto_hide(self, seconds: float = AUTO_HIDE_SECONDS) -> None:
        if self._timer is not None:
            self._timer.invalidate()
        self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            seconds, False, self._auto_hide
        )

    def _auto_hide(self, timer: Any) -> None:
        self._timer = None
        hovered = NSPointInRect(NSEvent.mouseLocation(), self._panel.frame())
        typing = self._panel.isKeyWindow() and bool(self._field.stringValue())
        if hovered or typing or self._busy:
            self._arm_auto_hide(RECHECK_SECONDS)
        else:
            self.hide()


def _label(text: str, font: Any, color: Any) -> Any:
    label = NSTextField.wrappingLabelWithString_(text)
    label.setFont_(font)
    label.setTextColor_(color)
    label.setSelectable_(True)
    label.setPreferredMaxLayoutWidth_(WIDTH - 2 * PADDING)
    return label
