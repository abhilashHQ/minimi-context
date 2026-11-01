"""Client for minimi's MCP connector: JSON-RPC 2.0 over newline-delimited stdio."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import IO, Any, Callable, NoReturn, Protocol

from .store import Store

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
BUDGET_WINDOW = timedelta(days=7)

_RESULT_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+Result[ \t]+\d+[ \t]*#*[ \t]*$")
_SOURCE_PREFIX = "- **Source:**"
_QUOTA = re.compile(
    r"\b(?:quota|limit)\s+(?:exceeded|reached|exhausted)\b|\brate[- ]limit|\btoo many requests\b",
    re.IGNORECASE,
)
_QUOTA_MAX_CHARS = 300


class MinimiError(Exception):
    """minimi failed: tool error, quota message, protocol error, timeout or dead process."""


class BudgetExceeded(Exception):
    """The weekly MCP call budget is spent. Callers continue without new data."""


@dataclass(frozen=True)
class ToolResult:
    text: str
    next_cursor: str | None


class MinimiClient(Protocol):
    def search_memory(
        self,
        query: str,
        time_range: Mapping[str, Any],
        source_app: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> ToolResult: ...

    def get_latest_context(self, hours: int, limit: int) -> ToolResult: ...

    def list_active_threads(self, window: str) -> ToolResult: ...


def look_back(hours: int) -> dict[str, Any]:
    return {"last": {"hours": hours}}


def split_results(text: str) -> list[str]:
    """Split search output into one block per result without parsing any fields.

    A block starts at a `Result <number>` heading whose next non-blank line starts with
    `- **Source:**`. Fence state is ignored: context is raw screen text and can hold a
    stray ``` line. The preamble before the first block is dropped. If no boundary is
    found the format is unknown and nothing is returned, so no unrecognised text can
    reach the privacy gate.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _is_result_start(lines, i)]
    if not starts:
        if text.strip():
            log.warning("mcp output has no result boundaries chars=%d", len(text))
        return []
    ends = starts[1:] + [len(lines)]
    return ["\n".join(lines[start:end]).strip() for start, end in zip(starts, ends)]


def _is_result_start(lines: list[str], index: int) -> bool:
    if not _RESULT_HEADING.match(lines[index]):
        return False
    following = next((line for line in lines[index + 1 :] if line.strip()), "")
    return following.lstrip().startswith(_SOURCE_PREFIX)


class CallBudget:
    """Records every tools/call and refuses calls once the weekly budget is reached."""

    def __init__(self, store: Store, weekly_limit: int, clock: Callable[[], datetime]) -> None:
        self._store = store
        self._limit = weekly_limit
        self._clock = clock

    def spend(self, tool: str) -> None:
        now = self._clock()
        if self._limit and self._store.mcp_calls_since(now - BUDGET_WINDOW) >= self._limit:
            log.warning("mcp budget exceeded limit=%d tool=%s", self._limit, tool)
            raise BudgetExceeded(f"weekly MCP budget of {self._limit} calls reached")
        self._store.record_mcp_call(tool, now)


class StdioMinimiClient:
    """One minimi process per batch: started on the first call, terminated on close."""

    def __init__(
        self,
        command: Sequence[str],
        env: Mapping[str, str],
        budget: CallBudget,
        timeout: float = 30.0,
    ) -> None:
        self._command = list(command)
        self._env = {**os.environ, **env}
        self._budget = budget
        self._timeout = timeout
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._next_id = 1
        self._broken = False

    def __enter__(self) -> StdioMinimiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def search_memory(
        self,
        query: str,
        time_range: Mapping[str, Any],
        source_app: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> ToolResult:
        arguments: dict[str, Any] = {"query": query, "time_range": dict(time_range), "limit": limit}
        if source_app is not None:
            arguments["source_app"] = source_app
        if cursor is not None:
            arguments["cursor"] = cursor
        return self._call_tool("search_memory", arguments)

    def get_latest_context(self, hours: int, limit: int) -> ToolResult:
        return self._call_tool("get_latest_context", {"hours": hours, "limit": limit})

    def list_active_threads(self, window: str) -> ToolResult:
        return self._call_tool("list_active_threads", {"window": window})

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError):
            _stream(proc.stdin).close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self._broken:
            raise MinimiError("minimi process is unavailable for this batch")
        self._budget.spend(name)
        self._ensure_started()
        started = time.monotonic()
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        tool_result = _tool_result(result)
        log.info(
            "mcp call tool=%s chars=%d duration_ms=%d",
            name,
            len(tool_result.text),
            (time.monotonic() - started) * 1000,
        )
        return tool_result

    def _ensure_started(self) -> None:
        if self._proc is not None:
            return
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=self._env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self._broken = True
            raise MinimiError(f"cannot start minimi: {type(exc).__name__}") from None
        threading.Thread(
            target=_pump, args=(_stream(self._proc.stdout), self._lines), daemon=True
        ).start()
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "person-context", "version": "0.1.0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                line = self._lines.get(timeout=max(remaining, 0.0))
            except queue.Empty:
                self._fail(f"timeout after {self._timeout:g}s on {method}")
            if line is None:
                self._fail(f"minimi exited during {method}")
            try:
                message = json.loads(line)
            except ValueError:
                log.debug("mcp skipped non-JSON stdout line")
                continue
            if (
                not isinstance(message, dict)
                or "method" in message
                or message.get("id") != request_id
            ):
                continue
            if "error" in message:
                error = message["error"]
                code = error.get("code") if isinstance(error, dict) else None
                raise MinimiError(f"{method} returned JSON-RPC error {code}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise MinimiError(f"{method} returned no result object")
            return result

    def _send(self, message: dict[str, Any]) -> None:
        assert self._proc is not None
        try:
            stdin = _stream(self._proc.stdin)
            stdin.write(json.dumps(message) + "\n")
            stdin.flush()
        except OSError:
            self._fail("minimi stdin is closed")

    def _fail(self, reason: str) -> NoReturn:
        self._broken = True
        log.warning("mcp failure: %s", reason)
        self.close()
        raise MinimiError(reason)


def _stream(stream: IO[str] | None) -> IO[str]:
    if stream is None:
        raise MinimiError("minimi pipe is missing")
    return stream


def _pump(stream: IO[str], lines: queue.Queue[str | None]) -> None:
    try:
        for line in stream:
            lines.put(line)
    except (OSError, ValueError):
        pass
    finally:
        lines.put(None)


def _tool_result(result: Mapping[str, Any]) -> ToolResult:
    if result.get("isError"):
        raise MinimiError("tool returned an error")
    content = result.get("content")
    parts = content if isinstance(content, list) else []
    text = "\n".join(
        part["text"]
        for part in parts
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    )
    if len(text) <= _QUOTA_MAX_CHARS and _QUOTA.search(text):
        raise MinimiError("tool returned a quota message")
    structured = result.get("structuredContent")
    cursor = None
    if isinstance(structured, dict):
        found = structured.get("nextCursor", structured.get("next_cursor"))
        cursor = found if isinstance(found, str) and found else None
    return ToolResult(text=text, next_cursor=cursor)
