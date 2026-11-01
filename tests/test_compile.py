from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeLLM, FakeMinimi, FakeMinimiFactory
from helpers import T0, at, make_config, minimi_page, minimi_result

from person_context.compile import (
    CompileLocked,
    compile_all,
    compile_lock,
    parse_timestamp,
)
from person_context.config import Config
from person_context.llm import LLMUnavailable
from person_context.minimi import MinimiError
from person_context.models import Alias, normalize
from person_context.prompts import COMPILE_SCHEMA
from person_context.store import Store

ASHA_CONTEXT = (
    "From: Asha Rao <asha@example.com>\nAsha Rao: We'd like 40 units, delivered by Friday."
)
KARAN_CONTEXT = "Karan Mehta: my salary review is next week, keep it quiet"
ASHA_RESULT = minimi_result(ASHA_CONTEXT, captured="2026-09-28 14:02")
KARAN_RESULT = minimi_result(KARAN_CONTEXT, source="WhatsApp", captured="2026-09-28 15:10")
PAGE = minimi_page(ASHA_RESULT, KARAN_RESULT)


def extraction(
    *events: dict[str, str], proposals: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    return {"events": list(events), "alias_proposals": proposals or []}


def event(quote: str, **extra: str) -> dict[str, str]:
    item = {
        "text": "Asha wants 40 units by Friday.",
        "source_quote": quote,
        "occurred_at": "2026-09-28T14:02:00Z",
        "source_app": "Mail",
        "source_thread": "Order",
        "speaker": "them",
    }
    item.update(extra)
    return item


GOOD = event("We'd like 40 units, delivered by Friday.")


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_config(tmp_path / "context.db")


@pytest.fixture
def store(cfg: Config) -> Store:
    store = Store.open(cfg.db_path)
    store.create_person("Asha Rao", T0)
    return store


def run(
    store: Store,
    cfg: Config,
    minimi: FakeMinimi,
    llm: FakeLLM,
    hours: float = 0,
    limit: int | None = None,
) -> Any:
    factory = FakeMinimiFactory(minimi, weekly_limit=limit)
    [result] = compile_all(store, factory, llm, cfg, at(hours))
    return result


def test_stores_grounded_events_and_advances_watermark(store: Store, cfg: Config) -> None:
    llm = FakeLLM(extraction(GOOD))
    result = run(store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), llm, hours=1)
    assert result.outcome == "stored" and result.events_added == 1
    assert result.blocks_passed == 1 and result.blocks_dropped == 1
    [stored] = store.timeline(1, 300)
    assert stored.source_quote == GOOD["source_quote"]
    assert stored.occurred_at == at(0).replace(month=9, day=28, hour=14, minute=2)
    assert stored.speaker == "them" and stored.source_app == "Mail"
    assert store.watermark(1) == at(1)
    assert llm.calls[0]["model"] == "compile-model"
    assert llm.calls[0]["schema"] is COMPILE_SCHEMA


def test_other_persons_block_never_reaches_llm(store: Store, cfg: Config) -> None:
    llm = FakeLLM(extraction(GOOD))
    run(store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), llm)
    assert "Karan" not in llm.prompts()
    assert "salary" not in llm.prompts()
    assert "Asha Rao: We'd like 40 units" in llm.prompts()


def test_idempotent(store: Store, cfg: Config) -> None:
    minimi = FakeMinimi({"Asha Rao": [PAGE]})
    assert run(store, cfg, minimi, FakeLLM(extraction(GOOD)), hours=1).events_added == 1
    rerun_llm = FakeLLM(extraction(event("We'd like 40 units")))
    second = run(store, cfg, minimi, rerun_llm, hours=2)
    assert second.outcome == "nothing_new" and rerun_llm.calls == []
    assert store.event_count(1) == 1
    assert store.watermark(1) == at(1)


def test_only_unseen_blocks_reach_llm_and_same_quote_is_not_duplicated(
    store: Store, cfg: Config
) -> None:
    run(store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), FakeLLM(extraction(GOOD)), hours=1)
    fresh = minimi_result(f"{ASHA_CONTEXT}\nAsha Rao: also, Friday works", captured="2026-09-30")
    llm = FakeLLM(extraction(GOOD))
    result = run(store, cfg, FakeMinimi({"Asha Rao": [minimi_page(ASHA_RESULT, fresh)]}), llm, 2)
    assert result.outcome == "stored" and result.blocks_passed == 1
    assert result.events_added == 0 and store.event_count(1) == 1
    assert "Friday works" in llm.prompts() and "2026-09-28 14:02" not in llm.prompts()
    assert store.watermark(1) == at(2)


def test_failed_run_leaves_blocks_unseen(store: Store, cfg: Config) -> None:
    minimi = FakeMinimi({"Asha Rao": [PAGE]})
    assert run(store, cfg, minimi, FakeLLM(LLMUnavailable("down"))).outcome == "failed"
    assert store.seen_block_hashes(1) == set()
    llm = FakeLLM(extraction(GOOD))
    assert run(store, cfg, minimi, llm, hours=1).events_added == 1
    assert len(llm.calls) == 1 and len(store.seen_block_hashes(1)) == 1


def test_fabricated_quote_is_dropped(store: Store, cfg: Config) -> None:
    fabricated = event("We'd like 400 units, delivered tomorrow.")
    result = run(
        store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), FakeLLM(extraction(GOOD, fabricated))
    )
    assert result.events_added == 1 and result.events_dropped == 1
    assert [e.source_quote for e in store.timeline(1, 300)] == [GOOD["source_quote"]]


def test_quote_from_a_gated_out_block_is_dropped(store: Store, cfg: Config) -> None:
    leaked = event("my salary review is next week")
    result = run(store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), FakeLLM(extraction(leaked)))
    assert result.events_added == 0 and result.events_dropped == 1


def test_empty_results_leave_watermark_and_skip_llm(store: Store, cfg: Config) -> None:
    llm = FakeLLM()
    for pages in ({}, {"Asha Rao": [minimi_page(KARAN_RESULT)]}):
        result = run(store, cfg, FakeMinimi(pages), llm)
        assert result.outcome == "nothing_new"
    assert llm.calls == []
    assert store.watermark(1) is None


def test_llm_failure_leaves_watermark(store: Store, cfg: Config) -> None:
    result = run(store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), FakeLLM(LLMUnavailable("down")))
    assert result.outcome == "failed" and result.error == "LLMUnavailable"
    assert store.watermark(1) is None and store.event_count(1) == 0


def test_minimi_failure_leaves_watermark(store: Store, cfg: Config) -> None:
    llm = FakeLLM()
    result = run(store, cfg, FakeMinimi(error=MinimiError("tool error")), llm)
    assert result.outcome == "failed" and llm.calls == []
    assert store.watermark(1) is None


def test_budget_exceeded(store: Store, cfg: Config) -> None:
    store.record_mcp_call("search_memory", T0)
    minimi = FakeMinimi({"Asha Rao": [PAGE]})
    llm = FakeLLM()
    result = run(store, cfg, minimi, llm, limit=1)
    assert result.outcome == "failed" and result.error == "BudgetExceeded"
    assert minimi.searches == [] and llm.calls == []
    assert store.watermark(1) is None


def test_first_run_looks_back_14_days(store: Store, cfg: Config) -> None:
    minimi = FakeMinimi()
    run(store, cfg, minimi, FakeLLM())
    assert minimi.searches[0]["time_range"] == {"last": {"hours": 14 * 24}}


def test_paging_alias_dedupe_and_look_back_from_watermark(store: Store, cfg: Config) -> None:
    store.save_compile(
        1,
        [],
        [
            Alias(1, "asha rao", "name", "proposed", "asha rao"),
            Alias(1, "asha@example.com", "email", "proposed", "x"),
            Alias(1, "Acme", "company", "proposed", "x"),
        ],
        at(-2.5),
    )
    store.set_alias_status(1, "asha@example.com", "confirmed")
    store.set_alias_status(1, "Acme", "rejected")
    store.save_compile(1, [], [Alias(1, "Asha R.", "name", "proposed", "Asha R.")], at(-2.5))
    minimi = FakeMinimi({"Asha Rao": [minimi_page(minimi_result(f"page {i}")) for i in range(4)]})
    run(store, cfg, minimi, FakeLLM())
    assert [(s["query"], s["cursor"]) for s in minimi.searches] == [
        ("Asha Rao", None),
        ("Asha Rao", "1"),
        ("Asha Rao", "2"),
        ("asha@example.com", None),
    ]
    assert minimi.searches[0]["time_range"] == {"last": {"hours": 4}}


@pytest.mark.parametrize(("elapsed", "hours"), [(0, 1), (0.25, 2), (1, 2), (6, 7), (6.01, 8)])
def test_look_back_adds_one_hour_of_overlap(
    store: Store, cfg: Config, elapsed: float, hours: int
) -> None:
    store.save_compile(1, [], [], at(-elapsed))
    minimi = FakeMinimi()
    run(store, cfg, minimi, FakeLLM())
    assert minimi.searches[0]["time_range"] == {"last": {"hours": hours}}


def test_alias_proposals_need_grounded_evidence(store: Store, cfg: Config) -> None:
    proposals = [
        {
            "value": "asha@example.com",
            "kind": "email",
            "evidence_quote": "From: Asha Rao <asha@example.com>",
        },
        {"value": "Acme Corp", "kind": "company", "evidence_quote": "Asha Rao works at Acme Corp"},
        {"value": "+1 555 0100", "kind": "phone", "evidence_quote": "We'd like 40 units"},
    ]
    result = run(
        store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), FakeLLM(extraction(GOOD, proposals=proposals))
    )
    assert result.proposals_added == 1
    statuses = {a.value: a.status for a in store.aliases(1)}
    assert statuses == {"Asha Rao": "confirmed", "asha@example.com": "proposed"}


EMAIL_ONLY = minimi_page(minimi_result("From: asha@example.com\nShipping address changed to Pune"))
EMAIL_PROPOSAL = {
    "value": "asha@example.com",
    "kind": "email",
    "evidence_quote": "<asha@example.com>",
}


def propose_email(store: Store, cfg: Config) -> None:
    llm = FakeLLM(extraction(GOOD, proposals=[EMAIL_PROPOSAL]))
    run(store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), llm, hours=1)
    assert {a.value: a.status for a in store.aliases(1)}["asha@example.com"] == "proposed"


def test_pending_alias_neither_searches_nor_opens_gate_until_confirmed(
    store: Store, cfg: Config
) -> None:
    propose_email(store, cfg)
    pages = {"Asha Rao": [EMAIL_ONLY], "asha@example.com": [EMAIL_ONLY]}

    minimi, llm = FakeMinimi(pages), FakeLLM()
    result = run(store, cfg, minimi, llm, hours=2)
    assert [s["query"] for s in minimi.searches] == ["Asha Rao"]
    assert result.outcome == "nothing_new" and llm.calls == []

    assert store.set_alias_status(1, "asha@example.com", "confirmed")
    minimi, llm = FakeMinimi(pages), FakeLLM(extraction())
    run(store, cfg, minimi, llm, hours=3)
    assert [s["query"] for s in minimi.searches] == ["Asha Rao", "asha@example.com"]
    assert "Shipping address changed to Pune" in llm.prompts()


def test_rejected_alias_stays_closed_and_is_not_proposed_again(store: Store, cfg: Config) -> None:
    propose_email(store, cfg)
    assert store.set_alias_status(1, "asha@example.com", "rejected")

    minimi, llm = FakeMinimi({"Asha Rao": [EMAIL_ONLY]}), FakeLLM()
    result = run(store, cfg, minimi, llm, hours=2)
    assert [s["query"] for s in minimi.searches] == ["Asha Rao"]
    assert result.outcome == "nothing_new" and llm.calls == []

    again = [
        {"value": "ASHA@example.com", "kind": "email", "evidence_quote": "From: asha@example.com"}
    ]
    minimi = FakeMinimi(
        {"Asha Rao": [minimi_page(minimi_result("Asha Rao\nFrom: asha@example.com"))]}
    )
    result = run(store, cfg, minimi, FakeLLM(extraction(proposals=again)), hours=3)
    assert result.proposals_added == 0
    assert {a.value: a.status for a in store.aliases(1)}["asha@example.com"] == "rejected"


def test_compile_lock_prevents_overlap(store: Store, cfg: Config) -> None:
    llm = FakeLLM()
    with compile_lock(cfg.db_path):
        with pytest.raises(CompileLocked):
            compile_all(store, FakeMinimiFactory(FakeMinimi()), llm, cfg, T0)
    assert (
        compile_all(store, FakeMinimiFactory(FakeMinimi()), llm, cfg, T0)[0].outcome
        == "nothing_new"
    )


def test_one_minimi_batch_for_all_people(store: Store, cfg: Config) -> None:
    store.create_person("Karan Mehta", T0)
    factory = FakeMinimiFactory(FakeMinimi())
    results = compile_all(store, factory, FakeLLM(), cfg, T0)
    assert [r.person_id for r in results] == [1, 2]
    assert factory.batches == 1


def test_parse_timestamp() -> None:
    assert parse_timestamp("2026-09-28T14:02:00Z") == at(0).replace(
        month=9, day=28, hour=14, minute=2
    )
    assert parse_timestamp("2026-09-28") == at(0).replace(month=9, day=28, hour=0)
    offset = parse_timestamp("2026-09-28T19:32:00+05:30")
    assert offset is not None and offset.astimezone(timezone.utc).hour == 14
    assert parse_timestamp("") is None
    assert parse_timestamp("last Tuesday") is None


def test_normalized_quotes_match_blocks(store: Store, cfg: Config) -> None:
    lrm = "\N{LEFT-TO-RIGHT MARK}"
    quote = f"{lrm}we'd LIKE 40 units,   delivered by friday."
    result = run(store, cfg, FakeMinimi({"Asha Rao": [PAGE]}), FakeLLM(extraction(event(quote))))
    assert result.events_added == 1
    assert normalize(store.timeline(1, 1)[0].source_quote) == normalize(GOOD["source_quote"])


def test_other_persons_result_never_reaches_llm_and_blocks_keep_metadata(
    store: Store, cfg: Config
) -> None:
    page = minimi_page(
        KARAN_RESULT,
        minimi_result(
            "# Notes\n## Asha Rao\n---\nAsha Rao: call me Monday", captured="2026-09-29 10:00"
        ),
        minimi_result("Priya Nair: rent is due", source="Messages"),
    )
    llm = FakeLLM(extraction())
    result = run(store, cfg, FakeMinimi({"Asha Rao": [page]}), llm)
    assert result.blocks_passed == 1 and result.blocks_dropped == 2
    prompt = llm.prompts()
    assert "Karan" not in prompt and "salary" not in prompt and "Priya" not in prompt
    assert "- **Source:** Mail" in prompt and "- **Captured:** 2026-09-29 10:00" in prompt
    assert "## Asha Rao" in prompt and "Asha Rao: call me Monday" in prompt


def test_unknown_format_yields_nothing_and_no_llm_call(store: Store, cfg: Config) -> None:
    unknown = f"Found 2 results\n\n---\n{ASHA_CONTEXT}\n---\n{KARAN_CONTEXT}\n"
    llm = FakeLLM()
    result = run(store, cfg, FakeMinimi({"Asha Rao": [unknown]}), llm)
    assert result.outcome == "nothing_new" and llm.calls == []
    assert store.watermark(1) is None
