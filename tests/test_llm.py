from __future__ import annotations

import io
import json
import logging
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from person_context import llm
from person_context.llm import GeminiClient, LLMError, LLMInvalidOutput, LLMUnavailable

KEY = "secret-test-key-123"
SCHEMA = {
    "type": "OBJECT",
    "properties": {"answer": {"type": "STRING"}, "ok": {"type": "BOOLEAN"}},
    "required": ["answer", "ok"],
}


def gemini_payload(value: Any) -> bytes:
    text = value if isinstance(value, str) else json.dumps(value)
    return json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode()


class FakeUrlopen:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.requests: list[Request] = []
        self.timeouts: list[float] = []

    def __call__(self, request: Request, timeout: float) -> io.BytesIO:
        self.requests.append(request)
        self.timeouts.append(timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return io.BytesIO(outcome)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(llm, "sleep", recorded.append)
    return recorded


def install(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> FakeUrlopen:
    fake = FakeUrlopen(outcomes)
    monkeypatch.setattr(llm, "urlopen", fake)
    return fake


def http_error(code: int) -> HTTPError:
    return HTTPError("https://example.invalid", code, "error", {}, None)


def test_request_shape(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    fake = install(monkeypatch, [gemini_payload({"answer": "a", "ok": True})])
    value = GeminiClient(KEY).generate_json("model-x", "system text", "prompt text", SCHEMA)
    assert value == {"answer": "a", "ok": True}
    request = fake.requests[0]
    assert request.full_url.endswith("/models/model-x:generateContent")
    assert KEY not in request.full_url
    assert request.get_header("X-goog-api-key") == KEY
    body = json.loads(request.data)
    assert body["generationConfig"] == {
        "responseMimeType": "application/json",
        "responseSchema": SCHEMA,
    }
    assert body["systemInstruction"]["parts"][0]["text"] == "system text"
    assert body["contents"][0]["parts"][0]["text"] == "prompt text"
    assert fake.timeouts == [60]
    assert sleeps == []


def test_retry_then_success(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    fake = install(
        monkeypatch,
        [
            http_error(429),
            http_error(503),
            URLError(socket.timeout("timed out")),
            gemini_payload({"answer": "a", "ok": False}),
        ],
    )
    assert GeminiClient(KEY).generate_json("m", "s", "p", SCHEMA) == {"answer": "a", "ok": False}
    assert len(fake.requests) == 4
    assert len(sleeps) == 3
    assert 1 <= sleeps[0] < 2 and 2 <= sleeps[1] < 3 and 4 <= sleeps[2] < 5
    assert KEY not in caplog.text


def test_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    install(monkeypatch, [http_error(500), socket.timeout(), http_error(429), http_error(503)])
    with pytest.raises(LLMUnavailable) as info:
        GeminiClient(KEY).generate_json("m", "s", "secret prompt", SCHEMA)
    assert len(sleeps) == 3
    assert KEY not in caplog.text and KEY not in str(info.value)
    assert "secret prompt" not in caplog.text


def test_client_error_is_not_retried(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    fake = install(monkeypatch, [http_error(400)])
    with pytest.raises(LLMError):
        GeminiClient(KEY).generate_json("m", "s", "p", SCHEMA)
    assert len(fake.requests) == 1 and sleeps == []


def test_connection_failure(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    install(monkeypatch, [URLError(ConnectionRefusedError())])
    with pytest.raises(LLMUnavailable):
        GeminiClient(KEY).generate_json("m", "s", "p", SCHEMA)


def test_invalid_json_retried_once_then_success(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    fake = install(
        monkeypatch, [gemini_payload("not json {"), gemini_payload({"answer": "a", "ok": True})]
    )
    assert GeminiClient(KEY).generate_json("m", "s", "p", SCHEMA)["answer"] == "a"
    assert len(fake.requests) == 2
    assert "not json" not in caplog.text and KEY not in caplog.text


@pytest.mark.parametrize(
    "bad",
    [
        gemini_payload("not json {"),
        gemini_payload({"answer": "a"}),
        gemini_payload({"answer": 3, "ok": True}),
        b"<html>",
        json.dumps({"candidates": []}).encode(),
    ],
)
def test_invalid_output_twice_fails(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], bad: bytes
) -> None:
    fake = install(monkeypatch, [bad, bad])
    with pytest.raises(LLMInvalidOutput):
        GeminiClient(KEY).generate_json("m", "s", "p", SCHEMA)
    assert len(fake.requests) == 2


def test_key_not_in_repr() -> None:
    assert KEY not in repr(GeminiClient(KEY))


def test_check_schema_enum_and_arrays() -> None:
    schema = {
        "type": "ARRAY",
        "items": {"type": "STRING", "enum": ["self", "them"]},
    }
    llm.check_schema(["self", "them"], schema)
    with pytest.raises(llm.InvalidOutput):
        llm.check_schema(["self", "nobody"], schema)
    with pytest.raises(llm.InvalidOutput):
        llm.check_schema({"a": 1}, schema)
