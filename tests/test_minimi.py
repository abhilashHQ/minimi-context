from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from helpers import T0, at, minimi_page, minimi_result

from person_context.minimi import (
    BudgetExceeded,
    CallBudget,
    MinimiError,
    StdioMinimiClient,
    ToolResult,
    look_back,
    split_results,
)
from person_context.store import Store

SERVER = str(Path(__file__).with_name("fake_minimi_server.py"))


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.open(tmp_path / "context.db")


def client(store: Store, mode: str, *, limit: int = 0, timeout: float = 5.0) -> StdioMinimiClient:
    budget = CallBudget(store, limit, lambda: T0)
    return StdioMinimiClient(
        [sys.executable, SERVER], {"FAKE_MINIMI_MODE": mode}, budget, timeout=timeout
    )


def echo(result: ToolResult) -> Any:
    """The JSON the fake server echoes inside its single result's context fence."""
    [block] = split_results(result.text)
    return json.loads(block.split("```")[1])


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_success_one_process_per_batch(store: Store) -> None:
    with client(store, "ok") as minimi:
        first = minimi.search_memory("Asha Rao", look_back(24))
        second = minimi.get_latest_context(hours=2, limit=5)
        third = minimi.list_active_threads("24h")
    echoes = [echo(r) for r in (first, second, third)]
    assert len({e["pid"] for e in echoes}) == 1
    assert echoes[0]["arguments"] == {
        "query": "Asha Rao",
        "time_range": {"last": {"hours": 24}},
        "limit": 20,
    }
    assert echoes[1]["arguments"] == {"hours": 2, "limit": 5}
    assert echoes[2]["arguments"] == {"window": "24h"}
    assert first.next_cursor is None
    assert gone(echoes[0]["pid"])
    assert store.mcp_calls_since(T0) == 3


def test_cursor_and_source_app_are_passed(store: Store) -> None:
    with client(store, "paged") as minimi:
        first = minimi.search_memory("Asha", look_back(1))
        page = minimi.search_memory("Asha", look_back(1), source_app="Mail", cursor="page-2")
    assert first.next_cursor == "page-2"
    assert echo(page)["arguments"]["cursor"] == "page-2"
    assert echo(page)["arguments"]["source_app"] == "Mail"
    assert page.next_cursor is None


def test_noise_on_stdout_is_skipped(store: Store) -> None:
    with client(store, "noise") as minimi:
        result = minimi.search_memory("Asha", look_back(1))
    assert echo(result)["arguments"]["query"] == "Asha"


@pytest.mark.parametrize("mode", ["tool_error", "quota", "rpc_error"])
def test_tool_errors_raise(store: Store, mode: str) -> None:
    with client(store, mode) as minimi, pytest.raises(MinimiError):
        minimi.search_memory("Asha", look_back(1))


def test_timeout(store: Store) -> None:
    started = time.monotonic()
    with client(store, "hang", timeout=0.5) as minimi:
        with pytest.raises(MinimiError, match="timeout"):
            minimi.search_memory("Asha", look_back(1))
        with pytest.raises(MinimiError):
            minimi.search_memory("Asha", look_back(1))
    assert time.monotonic() - started < 10


def test_dead_process(store: Store) -> None:
    with client(store, "die") as minimi, pytest.raises(MinimiError, match="exited"):
        minimi.search_memory("Asha", look_back(1))


def test_missing_command(store: Store) -> None:
    budget = CallBudget(store, 0, lambda: T0)
    with StdioMinimiClient(["/nonexistent/minimi"], {}, budget) as minimi:
        with pytest.raises(MinimiError):
            minimi.search_memory("Asha", look_back(1))


def test_budget_exceeded_before_call(store: Store) -> None:
    store.record_mcp_call("search_memory", at(-24 * 8))
    store.record_mcp_call("search_memory", at(-1))
    with client(store, "ok", limit=2) as minimi:
        minimi.search_memory("Asha", look_back(1))
        with pytest.raises(BudgetExceeded):
            minimi.search_memory("Asha", look_back(1))
    assert store.mcp_calls_since(at(-24 * 7)) == 2


def test_budget_zero_is_unlimited(store: Store) -> None:
    for _ in range(30):
        store.record_mcp_call("search_memory", T0)
    with client(store, "ok", limit=0) as minimi:
        minimi.search_memory("Asha", look_back(1))


def test_split_real_format_one_block_per_result() -> None:
    page = minimi_page(
        minimi_result("Karan Mehta: lunch?", source="WhatsApp"),
        minimi_result("Asha Rao: send the deck", captured="2026-10-03 09:00"),
    )
    blocks = split_results(page)
    assert len(blocks) == 2
    assert blocks[0].startswith("### Result 1") and "Karan" in blocks[0]
    assert "Asha" not in blocks[0] and "Search Results" not in blocks[0]
    assert blocks[1].startswith("### Result 2")
    assert "- **Source:** Mail" in blocks[1] and "- **Captured:** 2026-10-03 09:00" in blocks[1]


def test_headings_and_rules_inside_a_fence_do_not_split() -> None:
    context = "# Inbox\n## Result 7\n### Result 8\n---\nAsha Rao: notes\n***"
    [block] = split_results(minimi_page(minimi_result(context)))
    assert "## Result 7" in block and "### Result 8" in block and "Asha Rao: notes" in block
    assert "- **Source:**" in block and "- **Captured:**" in block


def test_unknown_format_yields_nothing(caplog: pytest.LogCaptureFixture) -> None:
    text = "Found 2\n\n---\nAsha Rao said hi\n---\nKaran said bye\n## Two\nmore"
    assert split_results(text) == []
    assert "chars=" in caplog.text and "Asha" not in caplog.text
    assert split_results("   ") == []
