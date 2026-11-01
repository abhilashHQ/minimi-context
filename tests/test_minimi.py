from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
from helpers import T0, at

from person_context.minimi import (
    BudgetExceeded,
    CallBudget,
    MinimiError,
    StdioMinimiClient,
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
    echoes = [json.loads(r.text) for r in (first, second, third)]
    assert len({e["pid"] for e in echoes}) == 1
    assert echoes[0]["arguments"] == {
        "query": "Asha Rao",
        "time_range": {"last": {"hours": 24}},
        "limit": 20,
    }
    assert echoes[1]["arguments"] == {"hours": 2, "limit": 5}
    assert echoes[2]["arguments"] == {"window": "24h"}
    assert first.next_cursor == "page-2"
    assert gone(echoes[0]["pid"])
    assert store.mcp_calls_since(T0) == 3


def test_cursor_and_source_app_are_passed(store: Store) -> None:
    with client(store, "ok") as minimi:
        page = minimi.search_memory("Asha", look_back(1), source_app="Mail", cursor="page-2")
    assert json.loads(page.text)["arguments"]["cursor"] == "page-2"
    assert json.loads(page.text)["arguments"]["source_app"] == "Mail"
    assert page.next_cursor is None


def test_noise_on_stdout_is_skipped(store: Store) -> None:
    with client(store, "noise") as minimi:
        result = minimi.search_memory("Asha", look_back(1))
    assert json.loads(result.text)["arguments"]["query"] == "Asha"


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


def test_split_on_rules() -> None:
    text = "Found 2\n\n---\nAsha said hi\n---\nKaran said bye\n"
    assert split_results(text) == ["Found 2", "Asha said hi", "Karan said bye"]


def test_split_on_shallowest_heading() -> None:
    text = "## One\nAsha\n### detail\nmore\n## Two\nKaran"
    assert split_results(text) == ["## One\nAsha\n### detail\nmore", "## Two\nKaran"]


def test_split_failure_is_one_block() -> None:
    assert split_results("  just one result about Asha  ") == ["just one result about Asha"]
    assert split_results("   ") == []
