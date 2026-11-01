from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeLLM, FakeMinimi, FakeMinimiFactory
from helpers import LRM, T0, at, make_config, make_event, minimi_page, minimi_result

from person_context.config import Config
from person_context.llm import LLMUnavailable
from person_context.minimi import BudgetExceeded
from person_context.models import Alias, Event
from person_context.prompts import ASK_SCHEMA, CARD_SCHEMA, event_ref
from person_context.screen import ScreenContext, ScreenMessage
from person_context.service import NO_CONTEXT, UNKNOWN_PERSON, Service, describe_source
from person_context.store import Store

QUOTE = "Asha Rao: the budget is approved at 12,000"
EVENT = make_event(1, QUOTE, occurred_at=at(-48), text="Budget approved at 12,000.")
SCREEN = ScreenContext(
    "Asha Rao", (ScreenMessage("them", f"{LRM}Asha Rao, ok so what now?, 10:07 AM"),)
)


class Clock:
    def __init__(self) -> None:
        self.now = at(0)

    def __call__(self) -> Any:
        return self.now


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_config(tmp_path / "context.db")


@pytest.fixture
def store(cfg: Config) -> Store:
    store = Store.open(cfg.db_path)
    store.create_person("Asha Rao", T0)
    store.save_compile(1, [EVENT], [], at(-1))
    return store


def card_reply(*lines: dict[str, Any], show: bool = True) -> dict[str, Any]:
    return {"show": show, "lines": list(lines)}


def line(quote: str = "budget is approved at 12,000", event: Event = EVENT) -> dict[str, Any]:
    return {
        "text": "Budget was approved at 12,000.",
        "event_ids": [event_ref(event)],
        "quote": quote,
    }


def make_service(
    cfg: Config, llm: FakeLLM, minimi: FakeMinimi | None = None, **kwargs: Any
) -> tuple[Service, Clock, FakeMinimiFactory]:
    clock = Clock()
    factory = FakeMinimiFactory(minimi or FakeMinimi())
    service = Service(cfg, lambda: Store.open(cfg.db_path), llm, factory, clock=clock, **kwargs)
    return service, clock, factory


def card_calls(llm: FakeLLM) -> list[dict[str, Any]]:
    return [c for c in llm.calls if c["schema"] is CARD_SCHEMA]


def test_card_on_return(store: Store, cfg: Config) -> None:
    llm = FakeLLM(card_reply(line()))
    service, clock, factory = make_service(cfg, llm)
    card = service.on_screen(SCREEN)
    assert card is not None
    assert card.person.display_name == "Asha Rao"
    [shown] = card.lines
    assert shown.text == "Budget was approved at 12,000."
    assert describe_source(shown.source) == "“budget is approved at 12,000” · Mail · 29 Sep 2026"
    assert store.seen(1) == (at(0), at(0))
    assert factory.batches == 1  # freshness ran before the card
    [call] = llm.calls
    assert call["model"] == "answer-model"
    assert "ok so what now?" in call["prompt"] and QUOTE in call["prompt"]


def test_repeat_ticks_and_return_inside_gap_do_nothing(store: Store, cfg: Config) -> None:
    llm = FakeLLM(card_reply(line()))
    service, clock, _ = make_service(cfg, llm)
    assert service.on_screen(SCREEN) is not None
    for hours in (0.001, 0.5, 5.9):
        clock.now = at(hours)
        assert service.on_screen(SCREEN) is None
    assert len(llm.calls) == 1
    assert store.seen(1)[0] == at(5.9)


def test_return_after_gap_but_inside_cooldown(store: Store, cfg: Config) -> None:
    cfg = make_config(cfg.db_path, gap_hours=1.0, cooldown_hours=3.0)
    llm = FakeLLM(card_reply(line()), card_reply(line()))
    service, clock, _ = make_service(cfg, llm)
    assert service.on_screen(SCREEN) is not None
    clock.now = at(2)
    store.set_last_seen(1, at(0.5))
    assert service.on_screen(SCREEN) is None
    clock.now = at(4)
    store.set_last_seen(1, at(2.5))
    assert service.on_screen(SCREEN) is not None


def test_show_false_or_no_lines_shows_nothing(store: Store, cfg: Config) -> None:
    for reply in (card_reply(line(), show=False), card_reply()):
        store.set_last_seen(1, at(-100))
        service, _, _ = make_service(cfg, FakeLLM(reply))
        assert service.on_screen(SCREEN) is None
        assert store.seen(1)[1] is None


def test_fabricated_and_visible_lines_are_dropped(store: Store, cfg: Config) -> None:
    visible = make_event(1, "Asha Rao: ok so what now?", occurred_at=at(-1))
    store.save_compile(1, [visible], [], at(-1))
    reply = card_reply(
        line("budget is approved at 15,000"),
        line("ok so what now?", visible),
        {"text": "x", "event_ids": ["0" * 12], "quote": "budget is approved"},
    )
    service, _, _ = make_service(cfg, FakeLLM(reply))
    assert service.on_screen(SCREEN) is None
    assert store.seen(1)[1] is None


def test_at_most_four_lines(store: Store, cfg: Config) -> None:
    service, _, _ = make_service(cfg, FakeLLM(card_reply(*[line()] * 6)))
    card = service.on_screen(SCREEN)
    assert card is not None and len(card.lines) == 4


def test_untracked_chat_fires_nothing(store: Store, cfg: Config) -> None:
    llm = FakeLLM()
    service, _, factory = make_service(cfg, llm)
    assert service.on_screen(ScreenContext("Family group", ())) is None
    assert llm.calls == [] and factory.batches == 0


def test_unconfirmed_header_fires_nothing(store: Store, cfg: Config) -> None:
    store.save_compile(1, [], [Alias(1, "Ashu", "name", "proposed", "Ashu")], at(-1))
    llm = FakeLLM()
    service, _, _ = make_service(cfg, llm)
    assert service.on_screen(ScreenContext("Ashu", ())) is None
    assert llm.calls == []
    assert store.seen(1)[0] == at(0)


def test_retrieval_failure_logs_and_shows_nothing(
    store: Store, cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken:
        def __init__(self, store: Store) -> None:
            pass

        def retrieve(self, person_id: int, question: str, screen: Any = None) -> list[Event]:
            raise RuntimeError("index missing")

    service, _, _ = make_service(cfg, FakeLLM(), retriever=Broken)
    assert service.on_screen(SCREEN) is None
    assert "card failed person=1 error=RuntimeError" in caplog.text
    assert store.seen(1)[0] == at(0)


def test_llm_failure_shows_nothing(store: Store, cfg: Config) -> None:
    service, _, _ = make_service(cfg, FakeLLM(LLMUnavailable("down")))
    assert service.on_screen(SCREEN) is None
    assert store.seen(1)[1] is None


def test_freshness_failure_continues_with_stored_events(store: Store, cfg: Config) -> None:
    minimi = FakeMinimi(error=BudgetExceeded("spent"))
    llm = FakeLLM(card_reply(line()))
    service, _, _ = make_service(cfg, llm, minimi)
    assert service.on_screen(SCREEN) is not None
    assert store.watermark(1) == at(-1)


def test_freshness_adds_new_events_before_the_card(store: Store, cfg: Config) -> None:
    fresh = "Asha Rao: actually make it 14,000"
    minimi = FakeMinimi({"Asha Rao": [minimi_page(minimi_result(fresh))]})
    extraction = {
        "events": [
            {
                "text": "Budget is now 14,000.",
                "source_quote": fresh,
                "occurred_at": "",
                "source_app": "Mail",
                "source_thread": "",
                "speaker": "them",
            }
        ],
        "alias_proposals": [],
    }
    new_event = make_event(1, fresh)
    llm = FakeLLM(extraction, card_reply(line("make it 14,000", new_event)))
    service, _, _ = make_service(cfg, llm, minimi)
    card = service.on_screen(SCREEN)
    assert card is not None and card.lines[0].source.quote == "make it 14,000"
    assert store.watermark(1) == at(0)


def test_no_events_skips_llm(cfg: Config) -> None:
    store = Store.open(cfg.db_path)
    store.create_person("Asha Rao", T0)
    llm = FakeLLM()
    service, _, _ = make_service(cfg, llm)
    assert service.on_screen(SCREEN) is None
    assert llm.calls == []


def test_pending_alias_is_attached(store: Store, cfg: Config) -> None:
    store.save_compile(1, [], [Alias(1, "asha@example.com", "email", "proposed", "q")], at(-1))
    service, _, _ = make_service(cfg, FakeLLM(card_reply(line())))
    card = service.on_screen(SCREEN)
    assert card is not None and card.pending_alias is not None
    assert card.pending_alias.value == "asha@example.com"
    assert service.resolve_alias(1, "asha@example.com", confirm=True)
    assert not service.resolve_alias(1, "asha@example.com", confirm=False)


def test_retriever_is_swappable(store: Store, cfg: Config) -> None:
    seen: list[tuple[int, str, Any]] = []

    class OnlyOne:
        def __init__(self, store: Store) -> None:
            pass

        def retrieve(self, person_id: int, question: str, screen: Any = None) -> list[Event]:
            seen.append((person_id, question, screen))
            return [EVENT]

    service, _, _ = make_service(cfg, FakeLLM(card_reply(line())), retriever=OnlyOne)
    assert service.on_screen(SCREEN) is not None
    assert seen == [(1, "them: " + SCREEN.messages[0].text, SCREEN)]


def test_each_thread_gets_its_own_store(store: Store, cfg: Config) -> None:
    opened: list[int] = []

    def open_store() -> Store:
        opened.append(threading.get_ident())
        return Store.open(cfg.db_path)

    service = Service(cfg, open_store, FakeLLM(), FakeMinimiFactory(FakeMinimi()), clock=Clock())
    service.on_screen(ScreenContext("Nobody", ()))
    service.on_screen(ScreenContext("Nobody", ()))
    thread = threading.Thread(target=service.on_screen, args=(ScreenContext("Nobody", ()),))
    thread.start()
    thread.join()
    assert len(opened) == 2 and opened[0] != opened[1]


# Ask


def ask_reply(answer: str, *citations: tuple[Event, str]) -> dict[str, Any]:
    return {
        "answer": answer,
        "citations": [{"event_id": event_ref(e), "quote": q} for e, q in citations],
    }


def test_ask_about_person_on_screen(store: Store, cfg: Config) -> None:
    llm = FakeLLM(ask_reply("It was approved at 12,000.", (EVENT, "approved at 12,000")))
    service, _, _ = make_service(cfg, llm)
    result = service.ask("what budget did she approve?", SCREEN)
    assert result.status == "answered" and result.answer == "It was approved at 12,000."
    assert [s.quote for s in result.sources] == ["approved at 12,000"]
    [call] = [c for c in llm.calls if c["schema"] is ASK_SCHEMA]
    assert "what budget did she approve?" in call["prompt"]


def test_ask_names_another_person(store: Store, cfg: Config) -> None:
    store.create_person("Karan Mehta", T0)
    karan = make_event(2, "Karan Mehta: moving to Pune in March")
    store.save_compile(2, [karan], [], at(-1))
    llm = FakeLLM(ask_reply("He moves to Pune in March.", (karan, "moving to Pune")))
    service, _, _ = make_service(cfg, llm)
    result = service.ask("when is karan mehta moving?", SCREEN)
    assert result.person is not None and result.person.display_name == "Karan Mehta"
    assert result.status == "answered"


def test_ask_ambiguous_and_unknown(store: Store, cfg: Config) -> None:
    store.create_person("Karan Mehta", T0)
    llm = FakeLLM()
    service, _, _ = make_service(cfg, llm)
    ambiguous = service.ask("did Asha Rao meet Karan Mehta?", None)
    assert ambiguous.status == "ambiguous"
    assert "Asha Rao" in ambiguous.answer and "Karan Mehta" in ambiguous.answer
    unknown = service.ask("what does Priyanka want?", None)
    assert unknown.status == "unknown_person" and unknown.answer == UNKNOWN_PERSON
    untracked_screen = service.ask("what now?", ScreenContext("Family group", ()))
    assert untracked_screen.status == "unknown_person"
    assert llm.calls == []


def test_ask_without_surviving_citations(store: Store, cfg: Config) -> None:
    for reply in (
        ask_reply("Approved at 20,000.", (EVENT, "approved at 20,000")),
        ask_reply("No idea."),
    ):
        service, _, _ = make_service(cfg, FakeLLM(reply))
        result = service.ask("what budget?", SCREEN)
        assert result.status == "no_context" and result.answer == NO_CONTEXT
        assert result.sources == ()


def test_ask_failure_is_reported(store: Store, cfg: Config) -> None:
    service, _, _ = make_service(cfg, FakeLLM(LLMUnavailable("down")))
    result = service.ask("what budget?", SCREEN)
    assert result.status == "failed" and result.sources == ()


def test_ask_does_not_store_anything(store: Store, cfg: Config) -> None:
    llm = FakeLLM(ask_reply("Approved.", (EVENT, "approved")))
    service, _, _ = make_service(cfg, llm)
    service.ask("what budget?", SCREEN)
    assert store.event_count(1) == 1
    assert store.seen(1) == (None, None)
