from __future__ import annotations

import pytest

from baseline.build import build
from baseline.embed import unpack
from sme.store import repo

from .conftest import FakeEmbedder, FakeLLM


def test_build_one_call_per_conversation(tmp_path, scenario_a):
    llm = FakeLLM()
    db = tmp_path / "a.db"
    assert build(db, scenario_a, llm, FakeEmbedder()) == 9
    assert len(llm.prompts) == 3

    conn = repo.connect(db)
    rows = conn.execute(
        """SELECT mem.conversation_id, m.conversation_id AS msg_conv, v.dim, v.vector, v.model
           FROM memories mem
           JOIN messages m ON m.id = mem.message_id
           JOIN memory_vectors v ON v.memory_id = mem.id"""
    ).fetchall()
    assert len(rows) == 9
    for r in rows:
        assert r["conversation_id"] == r["msg_conv"]
        assert r["model"] == "fake-embedder"
        assert len(unpack(r["vector"])) == r["dim"] == FakeEmbedder.dim
    conn.close()


def test_prompt_shows_unknown_time(tmp_path, scenario_a):
    llm = FakeLLM()
    build(tmp_path / "a.db", scenario_a, llm, FakeEmbedder())
    sam_prompt = next(p for p in llm.prompts if "Conversation: Sam" in p)
    assert "time unknown | You: I'll call the landlord tomorrow about the flat" in sam_prompt


def test_bad_refs_are_skipped(tmp_path, scenario_a):
    class BadRefLLM:
        def generate_json(self, prompt, schema):
            return {
                "memories": [{"message_ref": "99", "text": "x"}, {"message_ref": "1", "text": "ok"}]
            }

    assert build(tmp_path / "a.db", scenario_a, BadRefLLM(), FakeEmbedder()) == 3


def test_build_refuses_to_append_twice(tmp_path, scenario_a):
    db = tmp_path / "a.db"
    build(db, scenario_a, FakeLLM(), FakeEmbedder())
    with pytest.raises(RuntimeError, match="already has memories"):
        build(db, scenario_a, FakeLLM(), FakeEmbedder())
