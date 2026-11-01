"""Configuration loaded and validated from the environment."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """A configuration variable is missing or invalid. The message names it."""


@dataclass(frozen=True)
class Config:
    gemini_api_key: str = field(repr=False)
    minimi_command: tuple[str, ...]
    minimi_env: Mapping[str, str] = field(repr=False)
    db_path: Path
    model_compile: str
    model_answer: str
    gap_hours: float
    cooldown_hours: float
    compile_hours: float
    mcp_weekly_budget: int


def load_config(environ: Mapping[str, str] | None = None) -> Config:
    env = os.environ if environ is None else environ
    return Config(
        gemini_api_key=_required(env, "GEMINI_API_KEY"),
        minimi_command=_command(env, "MINIMI_MCP_COMMAND"),
        minimi_env=_env_object(env, "MINIMI_MCP_ENV"),
        db_path=Path(env.get("PC_DB_PATH") or "~/.person-context/context.db").expanduser(),
        model_compile=_text(env, "PC_MODEL_COMPILE", "gemini-3.5-flash-lite"),
        model_answer=_text(env, "PC_MODEL_ANSWER", "gemini-3.6-flash"),
        gap_hours=_hours(env, "PC_GAP_HOURS", 6.0, allow_zero=True),
        cooldown_hours=_hours(env, "PC_COOLDOWN_HOURS", 2.0, allow_zero=True),
        compile_hours=_hours(env, "PC_COMPILE_HOURS", 6.0, allow_zero=False),
        mcp_weekly_budget=_budget(env, "PC_MCP_WEEKLY_BUDGET", 20),
    )


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required")
    return value


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    return env.get(name, "").strip() or default


def _command(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = _required(env, name)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ConfigError(f"{name} must be a JSON array of strings") from None
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(part, str) and part for part in value)
    ):
        raise ConfigError(f"{name} must be a non-empty JSON array of non-empty strings")
    return tuple(value)


def _env_object(env: Mapping[str, str], name: str) -> dict[str, str]:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ConfigError(f"{name} must be a JSON object of strings") from None
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ConfigError(f"{name} must be a JSON object with string values")
    return dict(value)


def _hours(env: Mapping[str, str], name: str, default: float, *, allow_zero: bool) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number of hours") from None
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = "zero or more" if allow_zero else "greater than zero"
        raise ConfigError(f"{name} must be {bound}")
    return value


def _budget(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number") from None
    if value < 0:
        raise ConfigError(f"{name} must be zero or more")
    return value
