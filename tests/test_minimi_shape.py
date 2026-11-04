from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from baseline.build import build
from baseline.minimi_shape import (
    INSTRUCTIONS,
    MinimiShapeTools,
    TimeRangeError,
    build_server,
    resolve_time_range,
)
from sme.store import repo

from .conftest import FakeEmbedder, FakeLLM

NOW_A = "2026-11-12T20:00:00+05:30"
NOW = repo.parse_time(NOW_A)


def call(server, name, args=None):
    result = asyncio.run(server.call_tool(name, args or {}))
    assert not result.is_error
    return result.structured_content


@pytest.fixture
def server(tmp_path, scenario_a):
    db = tmp_path / "a.db"
    build(db, scenario_a, FakeLLM(), FakeEmbedder())
    return build_server(MinimiShapeTools(db, NOW_A, FakeEmbedder(), "Asia/Kolkata"))


def test_tool_surface_matches_connector(server):
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert sorted(tools) == [
        "get_latest_context",
        "list_active_threads",
        "meeting_memory",
        "search_memory",
    ]
    assert set(tools["search_memory"].input_schema["properties"]) == {
        "query",
        "limit",
        "cursor",
        "time_range",
        "source_app",
    }
    assert set(tools["list_active_threads"].input_schema["properties"]) == {
        "window",
        "time_range",
        "apps",
        "limit",
    }
    assert set(tools["get_latest_context"].input_schema["properties"]) == {"hours", "apps", "limit"}
    assert "Orient before you drill in" in INSTRUCTIONS


def test_resolve_last():
    [(start, end)] = resolve_time_range({"last": {"days": 2, "hours": 3}}, NOW)
    assert end == NOW
    assert start == repo.parse_time("2026-11-10T17:00:00+05:30")


def test_resolve_day_with_between():
    [(start, end)] = resolve_time_range(
        {"day": "yesterday", "between": {"from": "09:00", "to": "10:30"}}, NOW
    )
    assert start == repo.parse_time("2026-11-11T09:00:00+05:30")
    assert end == repo.parse_time("2026-11-11T10:30:00+05:30")


def test_resolve_day_range_is_inclusive():
    windows = resolve_time_range({"day_range": {"from": "2026-11-10", "to": "2026-11-12"}}, NOW)
    assert [w[0].date().isoformat() for w in windows] == ["2026-11-10", "2026-11-11", "2026-11-12"]
    assert windows[-1][1] > datetime.fromisoformat("2026-11-12T23:59:00+05:30")


def test_resolve_rejects_empty_shape():
    with pytest.raises(TimeRangeError):
        resolve_time_range({"between": None, "foo": 1}, NOW)


def test_search_time_range_and_app_filter(server):
    out = call(
        server,
        "search_memory",
        {"query": "deck", "limit": 20, "time_range": {"day": "2026-11-12"}},
    )
    assert out["results"]
    assert {r["time"][:10] for r in out["results"]} == {"2026-11-12"}
    assert out["tz"] == "Asia/Kolkata"
    wa = call(server, "search_memory", {"query": "deck", "limit": 20, "source_app": "whatsapp"})
    assert {r["source_app"] for r in wa["results"]} == {"WhatsApp"}


def test_search_cursor_pages_through_results(server):
    first = call(server, "search_memory", {"query": "friday", "limit": 2})
    second = call(server, "search_memory", {"query": "friday", "limit": 2, "cursor": 2})
    assert first["next_cursor"] == 2
    assert not {r["memory"] for r in first["results"]} & {r["memory"] for r in second["results"]}


def test_bad_time_range_is_reported(server):
    out = call(server, "search_memory", {"query": "deck", "time_range": {"day": "someday"}})
    assert "bad day" in out["error"]


def test_list_active_threads_windows(server):
    today = call(server, "list_active_threads")
    assert [t["thread_title"] for t in today["threads"]] == ["Investor deck"]
    week = call(server, "list_active_threads", {"window": "this_week"})
    assert [t["thread_title"] for t in week["threads"]] == ["Investor deck", "Friday dinner", "Sam"]
    assert week["threads"][0]["preview"].startswith("You: ")


def test_get_latest_context_limit_and_apps(server):
    out = call(server, "get_latest_context", {"hours": 72, "limit": 1})
    assert [t["thread_title"] for t in out["threads"]] == ["Investor deck"]
    wa = call(server, "get_latest_context", {"hours": 72, "apps": ["WhatsApp"]})
    assert [t["source_app"] for t in wa["threads"]] == ["WhatsApp"]


def test_meeting_memory_finds_none(server):
    assert call(server, "meeting_memory", {"action": "list"})["meetings"] == []
