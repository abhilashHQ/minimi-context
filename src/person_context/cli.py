"""person-context command line: track | compile | ask | people | confirm | status | run."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import timedelta

from .config import Config, ConfigError, load_config
from .identity import resolve, track
from .minimi import BUDGET_WINDOW
from .service import AskResult, build_service, describe_source, utcnow
from .store import PersonExists, Store


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verbose = args.command in ("compile", "run")
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.command == "run":
        from .app import run  # pyobjc is only importable on macOS

        return int(run(cfg))
    handlers = {
        "track": _track,
        "compile": _compile,
        "ask": _ask,
        "people": _people,
        "confirm": _confirm,
        "status": _status,
    }
    return handlers[args.command](cfg, args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="person-context")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("track", help="track a person by name").add_argument("name")
    commands.add_parser("compile", help="compile new memories for everyone now")
    commands.add_parser("ask", help="ask about a tracked person").add_argument("question")
    commands.add_parser("people", help="list tracked people and their aliases")
    confirm = commands.add_parser("confirm", help="confirm or reject a proposed alias")
    confirm.add_argument("person_id", type=int)
    confirm.add_argument("alias")
    confirm.add_argument("--reject", action="store_true", help="reject instead of confirm")
    commands.add_parser("status", help="show counts, budget and compile state")
    commands.add_parser("run", help="run the macOS app")
    return parser


def _track(cfg: Config, args: argparse.Namespace) -> int:
    store = Store.open(cfg.db_path)
    try:
        person = track(store, args.name, utcnow())
    except (PersonExists, ValueError) as exc:
        print(f"cannot track: {exc}", file=sys.stderr)
        return 1
    print(f"tracking {person.display_name} as person {person.id}")
    return 0


def _compile(cfg: Config, args: argparse.Namespace) -> int:
    results = build_service(cfg).compile()
    if results is None:
        print("a compile is already running", file=sys.stderr)
        return 1
    for result in results:
        detail = f" ({result.error})" if result.error else ""
        print(
            f"person {result.person_id}: {result.outcome}{detail},"
            f" {result.events_added} events added, {result.proposals_added} aliases proposed"
        )
    return 1 if any(r.outcome == "failed" for r in results) else 0


def _ask(cfg: Config, args: argparse.Namespace) -> int:
    result: AskResult = build_service(cfg).ask(args.question, None)
    print(result.answer)
    for source in result.sources:
        print(f"  - {describe_source(source)}")
    return 1 if result.status == "failed" else 0


def _people(cfg: Config, args: argparse.Namespace) -> int:
    store = Store.open(cfg.db_path)
    for person in store.list_people():
        print(f"{person.id}  {person.display_name}")
        for alias in store.aliases(person.id):
            print(f"     {alias.kind:<8} {alias.status:<10} {alias.value}")
    return 0


def _confirm(cfg: Config, args: argparse.Namespace) -> int:
    store = Store.open(cfg.db_path)
    if not resolve(store, args.person_id, args.alias, confirm=not args.reject):
        print(f"no proposed alias {args.alias!r} for person {args.person_id}", file=sys.stderr)
        return 1
    print("rejected" if args.reject else "confirmed")
    return 0


def _status(cfg: Config, args: argparse.Namespace) -> int:
    store = Store.open(cfg.db_path)
    now = utcnow()
    budget = str(cfg.mcp_weekly_budget) if cfg.mcp_weekly_budget else "unlimited"
    print(f"database: {cfg.db_path}")
    calls = store.mcp_calls_since(now - BUDGET_WINDOW)
    print(f"mcp calls in the last {BUDGET_WINDOW.days} days: {calls} of {budget}")
    for person in store.list_people():
        watermark = store.watermark(person.id)
        pending = sum(1 for a in store.aliases(person.id) if a.status == "proposed")
        compiled = "never" if watermark is None else _ago(now - watermark)
        print(
            f"person {person.id} {person.display_name}: {store.event_count(person.id)} events,"
            f" {pending} pending aliases, last compile {compiled}"
        )
    return 0


def _ago(delta: timedelta) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours:.1f} hours ago"
