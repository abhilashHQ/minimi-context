"""LLM client: Gemini generateContent over urllib with JSON-schema responses."""

from __future__ import annotations

import json
import logging
import random
import socket
import time
from collections.abc import Mapping
from time import sleep
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
BACKOFF_SECONDS = 1.0
_RETRY_CODES = frozenset({429, 500, 503})
_TIMEOUTS = (socket.timeout, TimeoutError)


class LLMError(Exception):
    """The LLM call failed. Callers show nothing rather than something unverified."""


class LLMUnavailable(LLMError):
    """Retries were exhausted or the service could not be reached."""


class LLMInvalidOutput(LLMError):
    """The response did not match the expected structure, even after one retry."""


class InvalidOutput(ValueError):
    pass


class LLMClient(Protocol):
    def generate_json(self, model: str, system: str, prompt: str, schema: Mapping[str, Any]) -> Any:
        """Return parsed JSON that has been validated against `schema`."""


def check_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate `value` against the subset of the Gemini schema used in prompts.py."""
    kind = str(schema.get("type", "")).upper()
    if kind == "OBJECT":
        if not isinstance(value, dict):
            raise InvalidOutput(f"{path} must be an object")
        properties: Mapping[str, Any] = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                raise InvalidOutput(f"{path}.{name} is missing")
        for name, sub in properties.items():
            if name in value:
                check_schema(value[name], sub, f"{path}.{name}")
    elif kind == "ARRAY":
        if not isinstance(value, list):
            raise InvalidOutput(f"{path} must be an array")
        for index, item in enumerate(value):
            check_schema(item, schema.get("items", {}), f"{path}[{index}]")
    elif kind == "STRING":
        if not isinstance(value, str):
            raise InvalidOutput(f"{path} must be a string")
        if "enum" in schema and value not in schema["enum"]:
            raise InvalidOutput(f"{path} is not an allowed value")
    elif kind == "BOOLEAN":
        if not isinstance(value, bool):
            raise InvalidOutput(f"{path} must be a boolean")
    elif kind == "INTEGER":
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidOutput(f"{path} must be an integer")
    elif kind == "NUMBER":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidOutput(f"{path} must be a number")
    else:
        raise InvalidOutput(f"{path} has unsupported schema type {kind!r}")


class GeminiClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def __repr__(self) -> str:
        return "GeminiClient()"

    def generate_json(self, model: str, system: str, prompt: str, schema: Mapping[str, Any]) -> Any:
        body = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                },
            }
        ).encode("utf-8")
        for attempt in (1, 2):
            payload = self._post(model, body)
            try:
                value = _response_json(payload)
                check_schema(value, schema)
            except InvalidOutput as exc:
                log.warning("llm invalid output model=%s attempt=%d: %s", model, attempt, exc)
                continue
            return value
        raise LLMInvalidOutput(f"{model} returned invalid output twice")

    def _post(self, model: str, body: bytes) -> bytes:
        request = Request(
            API_URL.format(model=model),
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        reason = ""
        for attempt in range(MAX_RETRIES + 1):
            started = time.monotonic()
            try:
                with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    payload: bytes = response.read()
            except HTTPError as exc:
                if exc.code not in _RETRY_CODES:
                    raise LLMError(f"{model} returned HTTP {exc.code}") from None
                reason = f"HTTP {exc.code}"
            except _TIMEOUTS:
                reason = "timeout"
            except URLError as exc:
                if not isinstance(exc.reason, _TIMEOUTS):
                    kind = type(exc.reason).__name__
                    raise LLMUnavailable(f"{model} unreachable: {kind}") from None
                reason = "timeout"
            else:
                log.info(
                    "llm call model=%s attempt=%d duration_ms=%d",
                    model,
                    attempt + 1,
                    (time.monotonic() - started) * 1000,
                )
                return payload
            if attempt == MAX_RETRIES:
                break
            delay = BACKOFF_SECONDS * 2**attempt + random.uniform(0, BACKOFF_SECONDS)
            log.warning(
                "llm retry model=%s attempt=%d reason=%s delay_s=%.1f",
                model,
                attempt + 1,
                reason,
                delay,
            )
            sleep(delay)
        raise LLMUnavailable(f"{model} unavailable after {MAX_RETRIES} retries: {reason}")


def _response_json(payload: bytes) -> Any:
    try:
        data = json.loads(payload)
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part["text"] for part in parts if "text" in part and not part.get("thought"))
        return json.loads(text)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise InvalidOutput(f"response is not JSON content: {type(exc).__name__}") from None
