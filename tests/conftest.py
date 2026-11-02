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


NOISE_COUNT = re.compile(r"Write exactly (\d+) new messages sent on (\d{4}-\d{2}-\d{2})")
NOISE_SENDERS = re.compile(r"^Senders: only (.+)\.$", re.MULTILINE)
NOISE_TIMES = re.compile(r"between (\d{2}):(\d{2}) and (\d{2}):(\d{2})")


class FakeNoiseLLM:
    """Answers noise prompts with clean, unique messages at random times in the window."""

    model = "fake-noise"

    def __init__(self, seed: int = 0) -> None:
        import random

        self.rng = random.Random(seed)
        self.prompts: list[str] = []
        self.counter = 0

    def clean(self, prompt: str) -> list[dict[str, str]]:
        n = int(NOISE_COUNT.search(prompt).group(1))
        senders = NOISE_SENDERS.search(prompt).group(1).split(", ")
        h1, m1, h2, m2 = map(int, NOISE_TIMES.search(prompt).groups())
        out = []
        for i in range(n):
            self.counter += 1
            minute = self.rng.randint(h1 * 60 + m1, h2 * 60 + m2)
            out.append(
                {
                    "sender": senders[i % len(senders)],
                    "time": f"{minute // 60:02d}:{minute % 60:02d}",
                    "body": f"That football highlight was great, clip number {self.counter}",
                }
            )
        return out

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        self.prompts.append(prompt)
        return {"messages": self.clean(prompt)}


@pytest.fixture(scope="session")
def scenario_c_corpus(tmp_path_factory) -> dict[str, Any]:
    """A full merged scenario C corpus built from fake noise. Do not mutate."""
    from corpus import noise, scenario_c

    out = tmp_path_factory.mktemp("noise") / "noise.json"
    planted = scenario_c.read_json(scenario_c.PLANTED_PATH)
    generated = noise.generate(FakeNoiseLLM(), out, planted, log=lambda line: None)
    return scenario_c.merge(planted, generated)
