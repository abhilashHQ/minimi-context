"""LLM access behind one small interface."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-3.5-flash-lite"
FREE_TIER_RPM = 15
MAX_RATE_LIMIT_RETRIES = 8


class LLMClient(Protocol):
    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        """Return the model's reply to `prompt`, parsed as JSON matching `schema`."""
        ...


class GeminiClient:
    """Gemini over google-genai. Reads the key from GEMINI_API_KEY.

    Calls are spaced to stay under `rpm` requests a minute, and a 429 waits for the
    retry delay the API returns before trying again. A 5xx backs off and retries.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0,
        rpm: int = FREE_TIER_RPM,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        self._client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        from google.genai import errors

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                return self._generate(prompt, schema)
            except errors.ServerError:
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(min(2**attempt * 5, 120))
            except errors.ClientError as exc:
                if exc.code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(retry_delay_seconds(exc.details) + 1)
        raise AssertionError("unreachable")

    def _generate(self, prompt: str, schema: dict[str, Any]) -> Any:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        if not response.text:
            raise RuntimeError(f"empty response from {self.model}")
        return json.loads(response.text)


def retry_delay_seconds(details: Any, default: float = 60.0) -> float:
    """Read the RetryInfo delay, like "40s", from a Gemini 429 error body."""
    match = re.search(r"'retryDelay': '([\d.]+)s'", repr(details))
    return float(match.group(1)) if match else default


class BedrockClient:
    """gpt-oss-120b on Bedrock through converse. Bedrock has no response schema here, so the
    schema goes into the prompt and the reply's outermost JSON value is parsed."""

    def __init__(self, model: str = "openai.gpt-oss-120b-1:0", region: str = "us-east-1") -> None:
        import boto3
        from botocore.config import Config

        self.model = model
        self._client = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(read_timeout=300, retries={"max_attempts": 8, "mode": "standard"}),
        )

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        text = f"{prompt}\n\nReply with JSON only, matching this JSON schema:\n{json.dumps(schema)}"
        reply = self._client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": text}]}],
            inferenceConfig={"maxTokens": 8000, "temperature": 0},
            additionalModelRequestFields={"reasoning_effort": "low"},
        )
        out = "".join(c.get("text", "") for c in reply["output"]["message"]["content"])
        starts = [i for i in (out.find("{"), out.find("[")) if i >= 0]
        if not starts:
            raise RuntimeError(f"no JSON in reply from {self.model}")
        start = min(starts)
        end = out.rfind("}" if out[start] == "{" else "]")
        return json.loads(out[start : end + 1])
