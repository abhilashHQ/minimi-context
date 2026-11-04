"""JSON calls to gpt-oss-120b on Bedrock, cached in the store so reruns cost nothing."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from harness.store import Store

MODEL = "openai.gpt-oss-120b-1:0"
REGION = "us-east-1"


class LLMError(Exception):
    pass


@dataclass
class Stats:
    calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def parse_json(text: str) -> dict[str, Any]:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise LLMError("no JSON object in reply")
    try:
        return json.loads(text[a : b + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"bad JSON: {exc}") from exc


class LLM:
    def __init__(self, store: Store, model: str = MODEL, region: str = REGION) -> None:
        self.store = store
        self.model = model
        self.region = region
        self._client = None
        self.stats = Stats()

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime", region_name=self.region,
                config=Config(read_timeout=300, retries={"max_attempts": 8, "mode": "standard"}),
            )
        return self._client

    def _raw(self, system: str, user: str) -> str:
        key = hashlib.sha256(f"{self.model}\0{system}\0{user}".encode()).hexdigest()
        row = self.store.db.execute(
            "SELECT response FROM llm_cache WHERE prompt_hash = ?", (key,)
        ).fetchone()
        if row:
            self.stats.cache_hits += 1
            return row["response"]
        kwargs: dict[str, Any] = dict(
            modelId=self.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": 6000, "temperature": 0},
        )
        if "gpt-oss" in self.model:
            kwargs["additionalModelRequestFields"] = {"reasoning_effort": "medium"}
        t0 = time.time()
        r = self.client.converse(**kwargs)
        self.stats.seconds += time.time() - t0
        self.stats.calls += 1
        tin, tout = r["usage"]["inputTokens"], r["usage"]["outputTokens"]
        self.stats.tokens_in += tin
        self.stats.tokens_out += tout
        text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
        self.store.db.execute(
            "INSERT OR REPLACE INTO llm_cache(prompt_hash, response, tokens_in, tokens_out)"
            " VALUES (?, ?, ?, ?)",
            (key, text, tin, tout),
        )
        return text

    def json(self, system: str, user: str) -> dict[str, Any]:
        """One call, and one retry asking for valid JSON when the first reply doesn't parse."""
        try:
            return parse_json(self._raw(system, user))
        except LLMError:
            try:
                return parse_json(self._raw(system, user + "\n\nReply with one valid JSON object only."))
            except LLMError as exc:
                self.stats.errors.append(str(exc))
                raise
