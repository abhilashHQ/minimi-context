"""Demo plumbing for Claude Desktop.

    python -m harness.demo reset      # fresh copies of both stores into runs/demo/
    python -m harness.demo config     # add both servers to Claude Desktop (quit the app first)
    python -m harness.demo restore    # put the config back from the backup
    python -m harness.demo live       # the live moment: Sam moves tomorrow's dinner, on both sides

The left connector, "minimi-today", is minimi's 4-tool connector shape over one memory per message.
The right one, "minimi-harness", is the harness. Both read the same fictional month (scenario D).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILT_HARNESS = ROOT / "runs" / "harness" / "d.db"
BUILT_BASELINE = ROOT / "runs" / "baseline_d" / "d.db"
DEMO = ROOT / "runs" / "demo"
DEMO_HARNESS = DEMO / "harness.db"
DEMO_BASELINE = DEMO / "baseline.db"
NOW = "2026-11-26T20:00:00+05:30"
CONFIG = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
BACKUP = CONFIG.with_name(CONFIG.name + ".bak-before-harness")
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
LIVE_AT = "2026-11-26T19:55:00+05:30"
LIVE = ["Sam: Can we push tomorrow's dinner to 10? Class is running late again",
        "You: Sure, 10 works"]


def reset() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    for src, dst in ((BUILT_HARNESS, DEMO_HARNESS), (BUILT_BASELINE, DEMO_BASELINE)):
        if not src.exists():
            sys.exit(f"missing {src}; build it first")
        try:
            for suffix in ("", "-wal", "-shm"):
                Path(str(dst) + suffix).unlink(missing_ok=True)
            shutil.copyfile(src, dst)
        except PermissionError:
            sys.exit(f"{dst} is in use. Quit Claude Desktop fully, then run reset again.")
        print(f"{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def servers() -> dict:
    env = {"HF_HUB_OFFLINE": "1"}
    return {
        "minimi-today": {"command": PYTHON, "args": ["-W", "ignore", "-m", "baseline.minimi_shape",
                                                     "--db", str(DEMO_BASELINE), "--now", NOW], "env": env},
        "minimi-harness": {"command": PYTHON, "args": ["-W", "ignore", "-m", "harness.server",
                                                       "--db", str(DEMO_HARNESS)], "env": env},
    }


def config() -> None:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not BACKUP.exists():
        shutil.copyfile(CONFIG, BACKUP)
        print(f"backup: {BACKUP}")
    data["mcpServers"] = servers()
    CONFIG.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote mcpServers {list(data['mcpServers'])} to {CONFIG}")


def restore() -> None:
    if not BACKUP.exists():
        sys.exit("no backup to restore")
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    data["mcpServers"] = json.loads(BACKUP.read_text(encoding="utf-8")).get("mcpServers", {})
    CONFIG.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"restored mcpServers {list(data['mcpServers'])}")


def live() -> None:
    from harness.cli import main as cli

    args = ["add", "--db", str(DEMO_HARNESS), "--thread", "sam", "--baseline-db", str(DEMO_BASELINE),
            "--at", LIVE_AT]
    for m in LIVE:
        args += ["--msg", m]
    cli(args)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="harness.demo")
    p.add_argument("action", choices=["reset", "config", "restore", "live"])
    a = p.parse_args(argv)
    {"reset": reset, "config": config, "restore": restore, "live": live}[a.action]()


if __name__ == "__main__":
    main()
