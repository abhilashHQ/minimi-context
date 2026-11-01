from __future__ import annotations

from sme.store import repo

A_ORDER = ["m6", "m7", "m1", "m2", "m3", "m4", "m5", "m8", "m9"]


def count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def fts_hits(conn, term: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?", (term,)
    ).fetchone()[0]


def test_schema_file_matches_brief():
    sql = repo.SCHEMA_PATH.read_text(encoding="utf-8")
    assert sql.count("CREATE TABLE") == 2
    assert "CREATE VIRTUAL TABLE messages_fts USING fts5(body, content='messages');" in sql


def test_load_scenario_a(conn, scenario_a):
    result = repo.load_scenario(conn, scenario_a)
    assert result.conversations_added == 3
    assert result.messages_added == 9
    assert count(conn, "conversations") == 3
    assert count(conn, "messages") == 9

    convs = {(r["app"], r["external_key"]): r["title"] for r in repo.conversations(conn)}
    assert convs == {
        ("whatsapp_web", "sam"): "Sam",
        ("gmail", "friday-dinner"): "Friday dinner",
        ("gmail", "investor-deck"): "Investor deck",
    }

    m7 = conn.execute("SELECT * FROM messages WHERE id = ?", (result.message_ids["m7"],)).fetchone()
    assert m7["sender_label"] == "You"
    assert m7["direction"] == "out"
    assert m7["sent_at"] == "2026-11-11T10:40:00+05:30"
    assert m7["sent_minute"] == "2026-11-11T10:40"
    assert m7["body_norm"] == "9:30 works. i'll book the table by friday noon."
    assert m7["first_observed_at"] == "2026-11-12T20:00:00+05:30"

    m8 = conn.execute("SELECT * FROM messages WHERE id = ?", (result.message_ids["m8"],)).fetchone()
    assert (m8["sender_label"], m8["direction"]) == ("Priya", "in")


def test_date_source_by_app(conn, scenario_a):
    repo.load_scenario(conn, scenario_a)
    rows = conn.execute(
        """SELECT c.app, m.date_source, m.body FROM messages m
           JOIN conversations c ON c.id = m.conversation_id"""
    ).fetchall()
    for r in rows:
        if r["app"] == "gmail":
            assert r["date_source"] == "absolute"
        elif r["body"].startswith("I'll call the landlord"):
            assert r["date_source"] == "unknown"
        else:
            assert r["date_source"] == "separator"


def test_message_id_formula(conn, scenario_a):
    result = repo.load_scenario(conn, scenario_a)
    row = conn.execute(
        "SELECT * FROM messages WHERE id = ?", (result.message_ids["m1"],)
    ).fetchone()
    expected = repo.message_id(
        repo.conversation_id("whatsapp_web", "sam"),
        "Sam",
        "dinner this friday at 8?",
        "2026-11-09T19:05",
        1,
    )
    assert row["id"] == expected
    assert row["occurrence"] == 1


def test_body_norm_collapses_whitespace_and_lowercases():
    assert repo.normalize_body("  Yes,\n 8\tWORKS  ") == "yes, 8 works"


def test_idempotent_load(conn, scenario_a):
    first = repo.load_scenario(conn, scenario_a)
    second = repo.load_scenario(conn, scenario_a)
    assert second.conversations_added == 0
    assert second.messages_added == 0
    assert second.message_ids == first.message_ids
    assert count(conn, "conversations") == 3
    assert count(conn, "messages") == 9
    assert fts_hits(conn, "deck") == 1
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('integrity-check')")


def test_ingest_order(conn, scenario_a):
    assert [m["label"] for m in scenario_a["messages"]] == A_ORDER
    result = repo.load_scenario(conn, scenario_a)
    ids = [r["id"] for r in conn.execute("SELECT id FROM messages ORDER BY rowid")]
    assert ids == [result.message_ids[label] for label in A_ORDER]


def test_m5_has_null_date(conn, scenario_a):
    result = repo.load_scenario(conn, scenario_a)
    m5 = conn.execute("SELECT * FROM messages WHERE id = ?", (result.message_ids["m5"],)).fetchone()
    assert m5["body"] == "I'll call the landlord tomorrow about the flat"
    assert m5["sent_at"] is None
    assert m5["sent_minute"] is None
    assert m5["date_source"] == "unknown"
    assert m5["sender_label"] == "You"


def test_m10_only_in_scenario_b(tmp_path, scenario_a, scenario_b):
    assert scenario_b["messages"][:-1] == scenario_a["messages"]
    assert scenario_b["conversations"] == scenario_a["conversations"]
    assert scenario_a["now"] == "2026-11-12T20:00:00+05:30"
    assert scenario_b["now"] == "2026-11-13T17:00:00+05:30"

    conn_a = repo.connect(tmp_path / "a.db")
    conn_b = repo.connect(tmp_path / "b.db")
    res_a = repo.load_scenario(conn_a, scenario_a)
    res_b = repo.load_scenario(conn_b, scenario_b)
    assert "m10" not in res_a.message_ids
    assert count(conn_a, "messages") == 9
    assert count(conn_b, "messages") == 10
    m10 = conn_b.execute(
        "SELECT * FROM messages WHERE id = ?", (res_b.message_ids["m10"],)
    ).fetchone()
    assert m10["body"] == "Here is the deck, attached."
    assert m10["sent_at"] == "2026-11-13T16:00:00+05:30"
    assert m10["sender_label"] == "You"
    assert {k: v for k, v in res_b.message_ids.items() if k != "m10"} == res_a.message_ids
    conn_a.close()
    conn_b.close()


def test_repeated_identical_message_gets_next_occurrence(conn):
    scenario = {
        "now": "2026-11-12T20:00:00+05:30",
        "conversations": [{"app": "whatsapp_web", "key": "x", "title": "X"}],
        "messages": [
            {
                "label": f"r{i}",
                "app": "whatsapp_web",
                "thread": "x",
                "sender": "Sam",
                "sent_at": "2026-11-09T19:05:00+05:30",
                "date_source": "separator",
                "body": "ok",
            }
            for i in (1, 2)
        ],
    }
    result = repo.load_scenario(conn, scenario)
    assert result.messages_added == 2
    occ = [r["occurrence"] for r in conn.execute("SELECT occurrence FROM messages ORDER BY rowid")]
    assert occ == [1, 2]
    assert repo.load_scenario(conn, scenario).messages_added == 0
