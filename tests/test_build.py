from __future__ import annotations

import re
from datetime import datetime, timedelta

import pytest

from baseline.build import build
from baseline.embed import unpack
from sme.store import repo

from .conftest import FakeEmbedder, FakeLLM

NUMBERED = re.compile(r"^\[(\d+)\] ", re.MULTILINE)
START = datetime(2026, 11, 2, 8, 0, tzinfo=repo.IST)
CONTEXT = re.compile(r"^\(context\) .+ \| \w+: (.+)$", re.MULTILINE)


def quiet(line):
    return None


def long_scenario(n: int, name: str = "long") -> dict:
    return {
        "scenario": name,
        "now": "2026-11-26T20:00:00+05:30",
        "conversations": [
            {"app": "whatsapp_web", "key": "big", "title": "Big"},
            {"app": "gmail", "key": "small", "title": "Small"},
        ],
        "messages": [
            {
                "label": f"b{i}",
                "app": "whatsapp_web",
                "thread": "big",
                "sender": "Sam" if i % 2 else "You",
                "sent_at": repo.to_iso(START + timedelta(minutes=10 * i)),
                "date_source": "separator",
                "body": f"message {i}",
            }
            for i in range(n)
        ]
        + [
            {
                "label": "s1",
                "app": "gmail",
                "thread": "small",
                "sender": "Dev Mehta",
                "sent_at": "2026-11-03T09:30:00+05:30",
                "date_source": "absolute",
                "body": "small one",
            }
        ],
    }


def memory_counts(db):
    conn = repo.connect(db)
    total = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    per_msg = conn.execute(
        "SELECT max(c) FROM (SELECT count(*) AS c FROM memories GROUP BY message_id)"
    ).fetchone()[0]
    vectors = conn.execute("SELECT count(*) FROM memory_vectors").fetchone()[0]
    messages = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    conn.close()
    return total, per_msg, vectors, messages


def test_build_one_call_per_small_conversation(tmp_path, scenario_a):
    llm = FakeLLM()
    db = tmp_path / "a.db"
    assert build(db, scenario_a, llm, FakeEmbedder(), log=quiet) == 9
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
    build(tmp_path / "a.db", scenario_a, llm, FakeEmbedder(), log=quiet)
    sam_prompt = next(p for p in llm.prompts if "Conversation: Sam" in p)
    assert "time unknown | You: I'll call the landlord tomorrow about the flat" in sam_prompt
    assert "Write exactly one memory for each numbered message." in sam_prompt
    assert "(context)" not in sam_prompt


def test_chunks_of_40_with_5_context(tmp_path, capsys):
    llm = FakeLLM()
    db = tmp_path / "long.db"
    assert build(db, long_scenario(95), llm, FakeEmbedder()) == 96
    big = [p for p in llm.prompts if "Conversation: Big" in p]
    assert [len(NUMBERED.findall(p)) for p in big] == [40, 40, 15]
    assert CONTEXT.findall(big[0]) == []
    assert CONTEXT.findall(big[1]) == [f"message {i}" for i in range(35, 40)]
    assert CONTEXT.findall(big[2]) == [f"message {i}" for i in range(75, 80)]
    assert "[1] " in big[1] and "| You: message 40" in big[1].split("[1] ")[1].splitlines()[0]
    assert memory_counts(db) == (96, 1, 96, 96)
    out = capsys.readouterr().out
    assert "[1/4] Big chunk 1: 40 messages, 40 memories" in out
    assert "[4/4] Small chunk 1: 1 messages, 1 memories" in out


def test_exactly_one_memory_per_message(tmp_path, scenario_a):
    class Messy:
        def __init__(self):
            self.prompts = []

        def generate_json(self, prompt, schema):
            self.prompts.append(prompt)
            refs = NUMBERED.findall(prompt)
            if len(self.prompts) <= 3:
                # Doubles ref 1, cites a ref that does not exist, skips the rest.
                return {
                    "memories": [
                        {"message_ref": "1", "text": "first"},
                        {"message_ref": "1", "text": "again"},
                        {"message_ref": "99", "text": "x"},
                    ]
                }
            return {"memories": [{"message_ref": r, "text": f"retry {r}"} for r in refs]}

    llm = Messy()
    db = tmp_path / "a.db"
    assert build(db, scenario_a, llm, FakeEmbedder(), log=quiet) == 9
    total, per_msg, _, messages = memory_counts(db)
    assert (total, per_msg, messages) == (9, 1, 9)
    conn = repo.connect(db)
    texts = sorted(r[0] for r in conn.execute("SELECT text FROM memories"))
    conn.close()
    assert texts.count("first") == 3 and "again" not in texts
    # Sam has 5 messages: the retry numbers only the 4 still missing.
    sam_retry = next(p for p in llm.prompts[3:] if "Conversation: Sam" in p)
    assert len(NUMBERED.findall(sam_retry)) == 4
    assert "(context)" in sam_retry


def test_fallback_when_model_keeps_skipping(tmp_path, scenario_a, capsys):
    class Skips:
        def generate_json(self, prompt, schema):
            return {"memories": []}

    db = tmp_path / "a.db"
    assert build(db, scenario_a, Skips(), FakeEmbedder(), log=quiet) == 9
    conn = repo.connect(db)
    texts = [r[0] for r in conn.execute("SELECT text FROM memories")]
    conn.close()
    assert "Priya wrote in Gmail Investor deck at 2026-11-12T11:00:00+05:30:" in " ".join(texts)
    assert "using its text" in capsys.readouterr().err


def test_resume_after_crash(tmp_path):
    scenario = long_scenario(95)
    db = tmp_path / "long.db"

    class Crash(FakeLLM):
        def generate_json(self, prompt, schema):
            if len(self.prompts) == 2:
                raise RuntimeError("quota")
            return super().generate_json(prompt, schema)

    with pytest.raises(RuntimeError, match="quota"):
        build(db, scenario, Crash(), FakeEmbedder(), log=quiet)
    assert memory_counts(db)[0] == 80

    llm = FakeLLM()
    lines = []
    assert build(db, scenario, llm, FakeEmbedder(), log=lines.append) == 16
    assert len(llm.prompts) == 2
    assert lines[0] == "resuming: 2/4 chunks already done"
    assert memory_counts(db) == (96, 1, 96, 96)

    again = FakeLLM()
    assert build(db, scenario, again, FakeEmbedder(), log=quiet) == 0
    assert again.prompts == []
    assert memory_counts(db) == (96, 1, 96, 96)


def test_refuses_a_different_scenario(tmp_path):
    db = tmp_path / "long.db"
    build(db, long_scenario(10), FakeLLM(), FakeEmbedder(), log=quiet)
    with pytest.raises(RuntimeError, match="different scenario"):
        build(db, long_scenario(11), FakeLLM(), FakeEmbedder(), log=quiet)


def test_refuses_memories_from_an_old_build(tmp_path, scenario_a):
    db = tmp_path / "a.db"
    build(db, scenario_a, FakeLLM(), FakeEmbedder(), log=quiet)
    conn = repo.connect(db)
    conn.execute("DELETE FROM build_meta")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="already has memories"):
        build(db, scenario_a, FakeLLM(), FakeEmbedder(), log=quiet)


def test_rebuild_flag_deletes_database(tmp_path, scenario_a, monkeypatch):
    import baseline.build as build_mod
    import baseline.embed as embed_mod
    import sme.llm.client as client_mod

    monkeypatch.setattr(client_mod, "GeminiClient", FakeLLM)
    monkeypatch.setattr(embed_mod, "SentenceTransformerEmbedder", FakeEmbedder)
    db = tmp_path / "a.db"
    build_mod.main(["--scenario", "a", "--db", str(db)])
    build_mod.main(["--scenario", "a", "--db", str(db)])
    assert memory_counts(db)[0] == 9
    build_mod.main(["--scenario", "b", "--db", str(db), "--rebuild"])
    assert memory_counts(db)[0] == 10


def test_scenario_c_build(tmp_path, scenario_c_corpus):
    llm = FakeLLM()
    db = tmp_path / "c.db"
    n = len(scenario_c_corpus["messages"])
    assert build(db, scenario_c_corpus, llm, FakeEmbedder(), log=quiet) == n
    assert max(len(NUMBERED.findall(p)) for p in llm.prompts) == 40
    assert memory_counts(db) == (n, 1, n, n)
