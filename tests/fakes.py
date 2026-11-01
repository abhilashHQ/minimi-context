"""Fakes for MinimiClient, LLMClient and the minimi batch factory."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Callable

from helpers import T0

from person_context.llm import check_schema
from person_context.minimi import CallBudget, ToolResult
from person_context.store import Store


class FakeMinimi:
    """Serves pages of markdown per query. Spends a real CallBudget when given one."""

    def __init__(
        self,
        pages: Mapping[str, list[str]] | None = None,
        error: Exception | None = None,
        budget: CallBudget | None = None,
    ) -> None:
        self.pages = dict(pages or {})
        self.error = error
        self.budget = budget
        self.searches: list[dict[str, Any]] = []

    def search_memory(
        self,
        query: str,
        time_range: Mapping[str, Any],
        source_app: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> ToolResult:
        if self.budget is not None:
            self.budget.spend("search_memory")
        self.searches.append({"query": query, "time_range": dict(time_range), "cursor": cursor})
        if self.error is not None:
            raise self.error
        pages = self.pages.get(query, [])
        index = int(cursor) if cursor else 0
        text = pages[index] if index < len(pages) else ""
        next_cursor = str(index + 1) if index + 1 < len(pages) else None
        return ToolResult(text, next_cursor)

    def get_latest_context(self, hours: int, limit: int) -> ToolResult:
        return ToolResult("", None)

    def list_active_threads(self, window: str) -> ToolResult:
        return ToolResult("", None)


class FakeMinimiFactory:
    """Opens one FakeMinimi batch per call, like StdioMinimiClient's one process per batch."""

    def __init__(self, minimi: FakeMinimi, weekly_limit: int | None = None) -> None:
        self.minimi = minimi
        self.weekly_limit = weekly_limit
        self.batches = 0

    @contextmanager
    def __call__(self, store: Store) -> Iterator[FakeMinimi]:
        self.batches += 1
        if self.weekly_limit is not None:
            self.minimi.budget = CallBudget(store, self.weekly_limit, lambda: T0)
        yield self.minimi


Handler = Callable[[str, str, str], Any]


class FakeLLM:
    """Records every request. Responses come from a handler or a queue of values."""

    def __init__(self, *responses: Any, handler: Handler | None = None) -> None:
        self.responses = list(responses)
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, model: str, system: str, prompt: str, schema: Mapping[str, Any]) -> Any:
        self.calls.append({"model": model, "system": system, "prompt": prompt, "schema": schema})
        if self.handler is not None:
            value = self.handler(model, system, prompt)
        else:
            value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        check_schema(value, schema)
        return value

    def prompts(self) -> str:
        return "\n".join(call["prompt"] for call in self.calls)
