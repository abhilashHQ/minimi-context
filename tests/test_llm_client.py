import pytest
from google.genai import errors

from sme.llm import client as llm


def rate_limited(delay="2s"):
    body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}],
        }
    }
    return errors.ClientError(429, body)


@pytest.fixture
def gemini(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    c = llm.GeminiClient(api_key="test", rpm=0)
    return c, sleeps


def test_429_waits_for_retry_delay_then_succeeds(gemini, monkeypatch):
    c, sleeps = gemini
    replies = [rate_limited("40s"), {"ok": True}]

    def fake(prompt, schema):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(c, "_generate", fake)
    assert c.generate_json("p", {}) == {"ok": True}
    assert sleeps == [41.0]


def test_429_gives_up_after_max_retries(gemini, monkeypatch):
    c, sleeps = gemini

    def fake(prompt, schema):
        raise rate_limited()

    monkeypatch.setattr(c, "_generate", fake)
    with pytest.raises(errors.ClientError):
        c.generate_json("p", {})
    assert len(sleeps) == llm.MAX_RATE_LIMIT_RETRIES


def test_other_client_errors_are_not_retried(gemini, monkeypatch):
    c, sleeps = gemini

    def fake(prompt, schema):
        raise errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})

    monkeypatch.setattr(c, "_generate", fake)
    with pytest.raises(errors.ClientError):
        c.generate_json("p", {})
    assert sleeps == []


def test_calls_are_spaced_by_rpm(monkeypatch):
    sleeps = []
    clock = iter([100.0, 100.0, 101.0, 101.0])
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))
    c = llm.GeminiClient(api_key="test", rpm=15)
    monkeypatch.setattr(c, "_generate", lambda prompt, schema: {})
    c._last_call = 100.0
    c.generate_json("p", {})
    assert sleeps == [pytest.approx(4.0)]


def test_server_errors_back_off_and_retry(gemini, monkeypatch):
    c, sleeps = gemini
    replies = [
        errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}}),
        {"ok": 1},
    ]

    def fake(prompt, schema):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(c, "_generate", fake)
    assert c.generate_json("p", {}) == {"ok": 1}
    assert sleeps == [5]
