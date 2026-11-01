"""NSApplication, polling, compile scheduler and hotkey. macOS only."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSEvent,
    NSEventMaskKeyDown,
    NSEventModifierFlagCommand,
    NSEventModifierFlagControl,
    NSEventModifierFlagOption,
    NSEventModifierFlagShift,
    NSTimer,
)
from PyObjCTools import AppHelper

from . import ax_macos
from .config import Config
from .models import Alias
from .screen import ScreenContext, parse_screen
from .service import ASK_FAILED, AskResult, Service, build_service
from .ui_panel import ContextPanel

log = logging.getLogger(__name__)

POLL_SECONDS = 1.0
DEBOUNCE_SECONDS = 1.5
_KEY_J = 38
_MODIFIERS = (
    NSEventModifierFlagCommand
    | NSEventModifierFlagShift
    | NSEventModifierFlagOption
    | NSEventModifierFlagControl
)
_HOTKEY = NSEventModifierFlagCommand | NSEventModifierFlagShift


class App:
    """Main-thread glue. Service calls run on worker threads, UI updates on the main thread."""

    def __init__(self, cfg: Config, service: Service) -> None:
        self._cfg = cfg
        self._service = service
        self._panel = ContextPanel(on_ask=self._ask, on_resolve_alias=self._resolve_alias)
        self._screen_worker = ThreadPoolExecutor(1, thread_name_prefix="screen")
        self._ask_worker = ThreadPoolExecutor(1, thread_name_prefix="ask")
        self._compile_worker = ThreadPoolExecutor(1, thread_name_prefix="compile")
        self._screen_task: Future[None] | None = None
        self._compile_task: Future[Any] | None = None
        # Owned by the screen worker thread.
        self._header: str | None = None
        self._header_since = 0.0
        # Written by the screen worker, read on the main thread for Ask.
        self._screen: ScreenContext | None = None
        self._monitors: list[Any] = []

    def start(self) -> None:
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(POLL_SECONDS, True, self._poll)
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            self._cfg.compile_hours * 3600, True, self._compile
        )
        self._monitors = [
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                NSEventMaskKeyDown, self._global_key
            ),
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                NSEventMaskKeyDown, self._local_key
            ),
        ]

    # Polling and the card

    def _poll(self, timer: Any) -> None:
        if self._screen_task is not None and not self._screen_task.done():
            return
        pid = ax_macos.frontmost_whatsapp_pid()
        self._screen_task = self._screen_worker.submit(_logged(self._tick), pid, time.monotonic())

    def _tick(self, pid: int | None, now: float) -> None:
        root = ax_macos.read_window(pid) if pid is not None else None
        screen = parse_screen(root) if root is not None else None
        if screen is None:
            self._header, self._screen = None, None
            return
        if screen.person_name != self._header:
            self._header, self._header_since, self._screen = screen.person_name, now, None
            return
        if now - self._header_since < DEBOUNCE_SECONDS:
            return
        self._screen = screen
        card = self._service.on_screen(screen)
        if card is None:
            return
        latest_root = ax_macos.read_window(pid) if pid is not None else None
        latest = parse_screen(latest_root) if latest_root is not None else None
        if latest is not None and latest.person_name == screen.person_name:
            AppHelper.callAfter(self._panel.show_card, card)

    def _compile(self, timer: Any) -> None:
        if self._compile_task is None or self._compile_task.done():
            self._compile_task = self._compile_worker.submit(_logged(self._service.compile))

    # Ask and alias confirmation

    def _ask(self, question: str) -> None:
        screen = self._screen
        future = self._ask_worker.submit(self._service.ask, question, screen)
        future.add_done_callback(self._deliver_answer)

    def _deliver_answer(self, future: Future[AskResult]) -> None:
        error = future.exception()
        if error is not None:
            log.error("ask failed error=%s", type(error).__name__)
            result = AskResult("failed", ASK_FAILED, None, ())
        else:
            result = future.result()
        AppHelper.callAfter(self._panel.show_answer, result)

    def _resolve_alias(self, alias: Alias, confirm: bool) -> None:
        self._ask_worker.submit(
            _logged(self._service.resolve_alias), alias.person_id, alias.value, confirm=confirm
        )

    # Hotkey: Cmd+Shift+J toggles the Ask panel

    def _is_hotkey(self, event: Any) -> bool:
        return bool(event.keyCode() == _KEY_J and (event.modifierFlags() & _MODIFIERS) == _HOTKEY)

    def _global_key(self, event: Any) -> None:
        if self._is_hotkey(event):
            self._toggle_ask()

    def _local_key(self, event: Any) -> Any:
        if self._is_hotkey(event):
            self._toggle_ask()
            return None
        return event

    def _toggle_ask(self) -> None:
        screen = self._screen
        self._panel.toggle_ask(screen.person_name if screen is not None else None)


def _logged(function: Callable[..., Any]) -> Callable[..., None]:
    """Worker tasks log failures by type and keep the app running."""

    def run(*args: Any, **kwargs: Any) -> None:
        try:
            function(*args, **kwargs)
        except Exception as exc:
            log.error("task %s failed error=%s", function.__name__, type(exc).__name__)

    return run


def run(cfg: Config) -> int:
    if not ax_macos.ensure_trusted():
        log.warning("accessibility access not granted; WhatsApp cannot be read until it is")
    application = NSApplication.sharedApplication()
    application.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    controller = App(cfg, build_service(cfg))
    controller.start()
    log.info("person-context running")
    AppHelper.runEventLoop(installInterrupt=True)
    return 0
