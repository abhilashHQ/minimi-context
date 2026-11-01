# minimi-context

A person context layer on top of minimi. It organizes the memories minimi already generates around people, then retrieves what matters at the moment you return to someone or ask.

Implementation contract: [SPEC.md](SPEC.md).

## Setup

Python 3.9 or later on macOS.

```
pip install -e '.[dev]'
export GEMINI_API_KEY=...
export MINIMI_MCP_COMMAND='["path/to/minimi-mcp", "--stdio"]'
```

Optional variables and their defaults are listed in SPEC.md section 4. The app needs Accessibility access (System Settings, Privacy & Security, Accessibility) to read WhatsApp.

## Commands

```
person-context track "Name"            # start tracking a person
person-context compile                 # pull new memories now
person-context ask "question"          # ask about a person named in the question
person-context people                  # people and their aliases
person-context confirm ID ALIAS [--reject]
person-context status                  # counts, MCP budget, last compile
person-context run                     # the macOS app; Cmd+Shift+J toggles Ask
```

## Checks

```
pytest -q
ruff check .
mypy --strict src/person_context --exclude 'ax_macos|ui_panel|app'
```
