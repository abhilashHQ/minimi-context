from __future__ import annotations

from pathlib import Path

import pytest

from person_context.config import ConfigError, load_config

BASE = {"GEMINI_API_KEY": "test-key", "MINIMI_MCP_COMMAND": '["minimi-mcp", "--stdio"]'}


def test_defaults() -> None:
    cfg = load_config(BASE)
    assert cfg.minimi_command == ("minimi-mcp", "--stdio")
    assert cfg.minimi_env == {}
    assert cfg.db_path == Path("~/.person-context/context.db").expanduser()
    assert cfg.model_compile == "gemini-3.5-flash-lite"
    assert cfg.model_answer == "gemini-3.6-flash"
    assert (cfg.gap_hours, cfg.cooldown_hours, cfg.compile_hours) == (6.0, 2.0, 6.0)
    assert cfg.mcp_weekly_budget == 20


def test_overrides_accept_decimals() -> None:
    cfg = load_config(
        {
            **BASE,
            "MINIMI_MCP_ENV": '{"MINIMI_TOKEN": "abc"}',
            "PC_DB_PATH": "/tmp/x.db",
            "PC_GAP_HOURS": "0.5",
            "PC_COOLDOWN_HOURS": "1.25",
            "PC_COMPILE_HOURS": "3",
            "PC_MCP_WEEKLY_BUDGET": "0",
        }
    )
    assert cfg.minimi_env == {"MINIMI_TOKEN": "abc"}
    assert cfg.db_path == Path("/tmp/x.db")
    assert (cfg.gap_hours, cfg.cooldown_hours, cfg.compile_hours) == (0.5, 1.25, 3.0)
    assert cfg.mcp_weekly_budget == 0


def test_api_key_not_in_repr() -> None:
    assert "test-key" not in repr(load_config(BASE))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GEMINI_API_KEY", ""),
        ("MINIMI_MCP_COMMAND", ""),
        ("MINIMI_MCP_COMMAND", "minimi-mcp"),
        ("MINIMI_MCP_COMMAND", "[]"),
        ("MINIMI_MCP_COMMAND", "[1]"),
        ("MINIMI_MCP_ENV", "[]"),
        ("MINIMI_MCP_ENV", '{"A": 1}'),
        ("PC_GAP_HOURS", "soon"),
        ("PC_GAP_HOURS", "-1"),
        ("PC_COOLDOWN_HOURS", "nan"),
        ("PC_COMPILE_HOURS", "0"),
        ("PC_MCP_WEEKLY_BUDGET", "1.5"),
        ("PC_MCP_WEEKLY_BUDGET", "-2"),
    ],
)
def test_invalid_values_name_the_variable(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        load_config({**BASE, name: value})


def test_missing_required() -> None:
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_config({"MINIMI_MCP_COMMAND": BASE["MINIMI_MCP_COMMAND"]})
    with pytest.raises(ConfigError, match="MINIMI_MCP_COMMAND"):
        load_config({"GEMINI_API_KEY": "k"})
