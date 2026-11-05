from __future__ import annotations

import asyncio

import pytest

from baseline.build import build
from baseline.server import BaselineTools, build_server

from .conftest import FakeEmbedder, FakeLLM

NOW_A = "2026-11-12T20:00:00+05:30"
NOW_B = "2026-11-13T17:00:00+05:30"


def make_tools(tmp_path, scenario, now):
    db = tmp_path / f"{scenario['scenario']}.db"
    build(db, scenario, FakeLLM(), FakeEmbedder())
    return BaselineTools(db, now, FakeEmbedder())


def call(server, name, args=None):
    result = asyncio.run(server.call_tool(name, args or {}))
    assert not result.is_error
    return result.structured_content


@pytest.fixture
def tools_a(tmp_path, scenario_a):
    return make_tools(tmp_path, scenario_a, NOW_A)


def test_exactly_three_tools(tools_a):
    server = build_server(tools_a)
    names = sorted(t.name for t in asyncio.run(server.list_tools()))
    assert names == ["get_latest_context", "list_active_threads", "search_memory"]


def test_search_memory(tools_a):
    out = call(build_server(tools_a), "search_memory", {"query": "send me the deck", "limit": 2})
    assert out["as_of"] == NOW_A
    assert len(out["results"]) == 2
    top = out["results"][0]
    assert top["source_app"] == "Gmail"
    assert top["thread_title"] == "Investor deck"
    assert top["time"] == "2026-11-12T11:00:00+05:30"
    assert top["message"]["sender"] == "Priya"
    assert top["message"]["text"] == "Can you send me the deck by Friday evening?"
    assert "deck" in top["memory"]
    assert set(top) == {"memory", "score", "source_app", "thread_title", "time", "message"}


def test_search_memory_default_limit(tools_a):
    out = call(build_server(tools_a), "search_memory", {"query": "friday"})
    assert len(out["results"]) == 5


def test_list_active_threads(tools_a):
    out = call(build_server(tools_a), "list_active_threads")
    assert out["as_of"] == NOW_A
    assert [
        (t["source_app"], t["thread_title"], t["latest_message_time"]) for t in out["threads"]
    ] == [
        ("Gmail", "Investor deck", "2026-11-12T11:30:00+05:30"),
        ("Gmail", "Friday dinner", "2026-11-11T10:40:00+05:30"),
        ("WhatsApp", "Sam", "2026-11-10T18:22:00+05:30"),
    ]


def test_unknown_time_thread_sorts_last(tmp_path, scenario_a):
    scenario = dict(scenario_a)
    scenario["scenario"] = "unknown"
    scenario["conversations"] = [
        {"app": "whatsapp_web", "key": "undated", "title": "Undated"},
        *scenario_a["conversations"],
    ]
    scenario["messages"] = [
        {
            "label": "u1",
            "app": "whatsapp_web",
            "thread": "undated",
            "sender": "Sam",
            "sent_at": None,
            "date_source": "unknown",
            "body": "No date on this one",
        },
        *scenario_a["messages"],
    ]
    tools = make_tools(tmp_path, scenario, NOW_A)
    threads = call(build_server(tools), "list_active_threads")["threads"]
    assert [t["thread_title"] for t in threads] == [
        "Investor deck",
        "Friday dinner",
        "Sam",
        "Undated",
    ]
    assert threads[-1]["latest_message_time"] is None


def test_get_latest_context_default_window(tools_a):
    out = call(build_server(tools_a), "get_latest_context")
    assert out["as_of"] == NOW_A
    assert out["window_hours"] == 48
    titles = [t["thread_title"] for t in out["threads"]]
    assert titles == ["Investor deck", "Friday dinner"]
    deck = out["threads"][0]["messages"]
    assert [(m["sender"], m["text"]) for m in deck] == [
        ("Priya", "Can you send me the deck by Friday evening?"),
        ("You", "Yes, I'll send it by 6 PM on Friday."),
    ]


def test_get_latest_context_returns_only_messages_inside_window(tools_a):
    out = call(build_server(tools_a), "get_latest_context", {"hours": 72})
    sam = next(t for t in out["threads"] if t["thread_title"] == "Sam")
    assert [m["text"] for m in sam["messages"]] == [
        "Brunch on Saturday at 9:30 too?",
        "Sure, see you then",
    ]


def test_scenario_b_tools(tmp_path, scenario_b):
    tools = make_tools(tmp_path, scenario_b, NOW_B)
    server = build_server(tools)
    threads = call(server, "list_active_threads")
    assert threads["as_of"] == NOW_B
    assert threads["threads"][0]["latest_message_time"] == "2026-11-13T16:00:00+05:30"
    ctx = call(server, "get_latest_context", {"hours": 24})
    assert ctx["as_of"] == NOW_B
    assert ctx["threads"][0]["messages"][-1]["text"] == "Here is the deck, attached."


def test_no_computed_status_fields(tools_a):
    server = build_server(tools_a)
    outputs = [
        call(server, "search_memory", {"query": "deck"}),
        call(server, "list_active_threads"),
        call(server, "get_latest_context", {"hours": 96}),
    ]
    text = repr(outputs).lower()
    for word in ("owed", "commitment", "status", "current", "superseded", "due"):
        assert f"'{word}" not in text
