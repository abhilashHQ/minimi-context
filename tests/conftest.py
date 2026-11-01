"""Fakes so tests run with no network and no API key."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pytest

from sme.store import repo

LINE = re.compile(r"^\[(\d+)\] (.+?) \| (.+)$", re.MULTILINE)


class FakeLLM:
    """Writes one memory per message line in the prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        self.prompts.append(prompt)
        return {
            "memories": [
                {"message_ref": ref, "text": f"{rest} (at {when})"}
                for ref, when, rest in LINE.findall(prompt)
            ]
        }


class FakeEmbedder:
    """Deterministic bag-of-words hashing into a small vector."""

    name = "fake-embedder"
    dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(vec)
        return out


@pytest.fixture
def scenario_a() -> dict[str, Any]:
    return repo.read_scenario("a")


@pytest.fixture
def scenario_b() -> dict[str, Any]:
    return repo.read_scenario("b")


@pytest.fixture
def conn(tmp_path):
    c = repo.connect(tmp_path / "store.db")
    yield c
    c.close()
