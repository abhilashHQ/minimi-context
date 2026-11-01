from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeLLM, FakeMinimi, FakeMinimiFactory
from helpers import T0, at, make_config, make_event

from person_context import cli
from person_context.models import Alias
from person_context.prompts import event_ref
from person_context.service import Service
from person_context.store import Store


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "context.db"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("MINIMI_MCP_COMMAND", '["minimi-mcp"]')
    monkeypatch.setenv("PC_DB_PATH", str(path))
    return path


def test_config_error_names_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("MINIMI_MCP_COMMAND", '["minimi-mcp"]')
    assert cli.main(["people"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_track_people_confirm_status(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["track", "Asha  Rao"]) == 0
    assert cli.main(["track", "asha rao"]) == 1
    assert cli.main(["track", "   "]) == 1
    store = Store.open(db)
    store.save_compile(1, [], [Alias(1, "asha@example.com", "email", "proposed", "q")], T0)
    store.save_compile(1, [], [Alias(1, "Acme", "company", "proposed", "q")], T0)
    capsys.readouterr()

    assert cli.main(["people"]) == 0
    out = capsys.readouterr().out
    assert "1  Asha Rao" in out and "proposed" in out and "asha@example.com" in out

    assert cli.main(["confirm", "1", "asha@example.com"]) == 0
    assert cli.main(["confirm", "1", "Acme", "--reject"]) == 0
    assert cli.main(["confirm", "1", "Acme"]) == 1
    assert {a.value: a.status for a in store.aliases(1)} == {
        "Asha Rao": "confirmed",
        "asha@example.com": "confirmed",
        "Acme": "rejected",
    }
    capsys.readouterr()

    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "mcp calls in the last 7 days: 0 of 20" in out
    assert "person 1 Asha Rao: 0 events, 0 pending aliases, last compile" in out


def test_compile_and_ask_commands(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store.open(db)
    store.create_person("Asha Rao", T0)
    event = make_event(1, "Asha Rao: the budget is approved", occurred_at=at(0))
    store.save_compile(1, [event], [], T0)
    llm = FakeLLM(
        {"answer": "Approved.", "citations": [{"event_id": event_ref(event), "quote": "approved"}]}
    )

    def fake_service(cfg: object) -> Service:
        cfg_ = make_config(db)
        return Service(cfg_, lambda: Store.open(db), llm, FakeMinimiFactory(FakeMinimi()))

    monkeypatch.setattr(cli, "build_service", fake_service)
    assert cli.main(["compile"]) == 0
    assert "person 1: nothing_new" in capsys.readouterr().out
    assert cli.main(["ask", "what did Asha Rao approve?"]) == 0
    out = capsys.readouterr().out
    assert "Approved." in out and "“approved” · Mail · 01 Oct 2026" in out
