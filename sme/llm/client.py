"""LLM access behind one small interface."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class LLMClient(Protocol):
    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        """Return the model's reply to `prompt`, parsed as JSON matching `schema`."""
        ...


class GeminiClient:
    """Gemini over google-genai. Reads the key from GEMINI_API_KEY."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        self._client = genai.Client(api_key=key)
        self.model = model

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        if not response.text:
            raise RuntimeError(f"empty response from {self.model}")
        return json.loads(response.text)
