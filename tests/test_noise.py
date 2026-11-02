from __future__ import annotations

import json
import re

import pytest

from corpus import noise, scenario_c
from corpus.check import noise_rule_violations

from .conftest import FakeNoiseLLM


@pytest.fixture
def planted():
    return scenario_c.read_json(scenario_c.PLANTED_PATH)


def quiet(log):
    return None


def test_day_plan_meets_targets_exactly():
    for conv in scenario_c.CONVERSATIONS:
        plan = noise.day_plan(conv)
        if conv.weekday_time:
            assert len(plan) == 19
            assert set(plan.values()) == {1}
            assert all(scenario_c.date.fromisoformat(d).weekday() < 5 for d in plan)
        elif conv.noise_target == 0:
            assert plan == {}
        else:
            assert sum(plan.values()) == conv.noise_target
            assert len(plan) == 25
            assert min(plan) == "2026-11-02" and max(plan) == "2026-11-26"
        assert noise.day_plan(conv) == plan


def test_day_plan_recent_noise():
    for key in scenario_c.RECENT_NOISE_KEYS:
        plan = noise.day_plan(scenario_c.BY_KEY[key])
        assert plan["2026-11-25"] + plan["2026-11-26"] >= 1


def test_drops_and_retries_bad_messages(planted):
    conv = scenario_c.BY_KEY["sam"]

    class BadThenGood(FakeNoiseLLM):
        def generate_json(self, prompt, schema):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return {
                    "messages": [
                        {"sender": "Sam", "time": "10:00", "body": "Dinner was good"},
                        {"sender": "You", "time": "10:05", "body": "I'll check the score"},
                        {"sender": "Kabir", "time": "10:10", "body": "Great match"},
                        {"sender": "Sam", "time": "03:00", "body": "Up late with a film"},
                        {"sender": "Sam", "time": "19:00", "body": "Close to a planted one"},
                        {"sender": "Sam", "time": "9am", "body": "Bad time format"},
                        {"sender": "You", "time": "11:00", "body": "What a goal"},
                        {"sender": "Sam", "time": "11:05", "body": "what  a GOAL"},
                    ]
                }
            return {"messages": self.clean(prompt)}

    llm = BadThenGood()
    windows = noise.quiet_windows(planted)[("sam", "2026-11-02")]
    assert windows == [(19 * 60 + 5 - 15, 19 * 60 + 5 + 15), (19 * 60 + 7 - 15, 19 * 60 + 7 + 15)]
    got = noise.generate_slot(llm, conv, "2026-11-02", 4, windows, [], set())

    assert len(got) == 4
    bodies = {m["body"] for m in got}
    assert "What a goal" in bodies
    for bad in ("Dinner was good", "I'll check the score", "Great match", "Up late with a film",
                "Close to a planted one", "Bad time format", "what  a GOAL"):  # fmt: skip
        assert bad not in bodies
    for m in got:
        assert noise_rule_violations(m["sender"], m["body"], conv.members) == []
    assert [m["time"] for m in got] == sorted(m["time"] for m in got)

    assert len(llm.prompts) == 2
    assert "Write exactly 4 new messages" in llm.prompts[0]
    retry = llm.prompts[1]
    assert "Write exactly 3 new messages" in retry
    assert "These were rejected" in retry
    assert "banned term 'dinner'" in retry
    assert 'You message has banned term "I\'ll"' in retry
    assert "sender 'Kabir' is not a member" in retry
    assert "time 03:00 is outside 08:00 to 23:45" in retry
    assert "time 19:00 is too close to another message" in retry
    assert "time '9am' is not HH:MM" in retry
    assert "repeats an earlier message" in retry
    assert "Do not use these times: 18:50 to 19:20, 18:52 to 19:22." in retry


def test_gives_up_after_max_attempts():
    class AlwaysBad:
        def generate_json(self, prompt, schema):
            return {"messages": [{"sender": "Sam", "time": "10:00", "body": "lunch?"}]}

    conv = scenario_c.BY_KEY["sam"]
    with pytest.raises(noise.NoiseGenerationError, match="only 0 of 2 messages passed after 3"):
        noise.generate_slot(AlwaysBad(), conv, "2026-11-03", 2, [], [], set(), max_attempts=3)


def test_last_day_ends_before_now():
    college = scenario_c.BY_KEY["college"]
    assert noise.slot_window(college, "2026-11-26") == ("08:00", "19:59")
    assert noise.slot_window(college, "2026-11-25") == ("08:00", "23:59")
    item = {"sender": "Kabir", "time": "20:00", "body": "Old hostel story"}
    assert noise.message_problems(college, "2026-11-26", item, [], set()) == [
        "time 20:00 is outside 08:00 to 19:59"
    ]


def test_fixed_schedule_stamps_sender_and_time():
    class Wrong:
        prompts: list[str] = []

        def generate_json(self, prompt, schema):
            self.prompts.append(prompt)
            return {
                "messages": [{"sender": "You", "time": "14:00", "body": "Payments team shipped"}]
            }

    llm = Wrong()
    got = noise.generate_slot(llm, scenario_c.BY_KEY["standup"], "2026-11-04", 1, [], [], set())
    assert got == [{"sender": "Dev Mehta", "time": "09:30", "body": "Payments team shipped"}]
    assert "Senders: only Dev Mehta." in llm.prompts[0]
    assert "between 09:30 and 09:30" in llm.prompts[0]


def test_generated_noise_meets_spec(tmp_path, planted):
    out = tmp_path / "noise.json"
    llm = FakeNoiseLLM()
    data = noise.generate(llm, out, planted, log=quiet)
    assert json.loads(out.read_text(encoding="utf-8")) == data
    assert data["model"] == "fake-noise"
    convs = data["conversations"]
    assert set(convs) == {c.key for c in scenario_c.CONVERSATIONS if c.noise_target}
    for key, slots in convs.items():
        conv = scenario_c.BY_KEY[key]
        assert sum(len(v) for v in slots.values()) == conv.noise_target or conv.weekday_time
        for day, msgs in slots.items():
            assert "2026-11-02" <= day <= "2026-11-26"
            for m in msgs:
                assert m["sender"] in conv.members
                if day == "2026-11-26":
                    assert m["time"] < "20:00"
                if conv.weekday_time:
                    assert m["time"] == conv.weekday_time
    for key in scenario_c.RECENT_NOISE_KEYS:
        assert convs[key]["2026-11-25"] or convs[key]["2026-11-26"]
    assert not scenario_c.missing_slots(data)


def test_resume_skips_done_slots(tmp_path, planted):
    out = tmp_path / "noise.json"

    class Crash(FakeNoiseLLM):
        def generate_json(self, prompt, schema):
            if len(self.prompts) == 30:
                raise RuntimeError("network down")
            return super().generate_json(prompt, schema)

    with pytest.raises(RuntimeError, match="network down"):
        noise.generate(Crash(), out, planted, log=quiet)
    saved = json.loads(out.read_text(encoding="utf-8"))
    done = {(k, d) for k, v in saved["conversations"].items() for d in v}
    assert done

    second = FakeNoiseLLM(seed=1)
    data = noise.generate(second, out, planted, log=quiet)
    for key, day in done:
        assert data["conversations"][key][day] == saved["conversations"][key][day]
    title_key = {c.title: c.key for c in scenario_c.CONVERSATIONS}
    asked = {
        (title_key[re.search(r"^Conversation: (.+) \(", p, re.M).group(1)],
         re.search(r"sent on (\S+),", p).group(1))
        for p in second.prompts
    }  # fmt: skip
    assert asked and not asked & done
    assert not scenario_c.missing_slots(data)

    third = FakeNoiseLLM()
    assert noise.generate(third, out, planted, log=quiet) == data
    assert third.prompts == []


def test_force_starts_over(tmp_path, planted):
    out = tmp_path / "noise.json"
    noise.generate(FakeNoiseLLM(), out, planted, only={"kapoor"}, log=quiet)
    assert set(json.loads(out.read_text(encoding="utf-8"))["conversations"]) == {"kapoor"}

    noise.generate(FakeNoiseLLM(), out, planted, only={"priya"}, log=quiet)
    assert set(json.loads(out.read_text(encoding="utf-8"))["conversations"]) == {"kapoor", "priya"}

    noise.generate(FakeNoiseLLM(), out, planted, force=True, only={"priya"}, log=quiet)
    assert set(json.loads(out.read_text(encoding="utf-8"))["conversations"]) == {"priya"}


def test_rejects_foreign_file(tmp_path, planted):
    out = tmp_path / "noise.json"
    out.write_text(json.dumps({"scenario": "a", "conversations": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a scenario C noise file"):
        noise.generate(FakeNoiseLLM(), out, planted, log=quiet)


def test_prompt_lists_rules_and_context():
    conv = scenario_c.BY_KEY["kapoor"]
    recent = [
        {"day": "2026-11-03", "sender": "Mr. Kapoor", "time": "10:00", "body": "Rent received"}
    ]
    prompt = noise.build_prompt(conv, "2026-11-07", 2, conv.members, [], recent, [])
    assert "sent on 2026-11-07, a weekend day." in prompt
    assert "Senders: only You, Mr. Kapoor." in prompt
    assert "rent receipts, building notices" in prompt
    assert "- 2026-11-03 10:00 Mr. Kapoor: Rent received" in prompt
    assert "cap table" in prompt and "by Sunday" in prompt
