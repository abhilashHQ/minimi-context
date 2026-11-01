# Spec: person context layer

This document is the contract. Build exactly this. If something here is impossible or contradictory, stop and say so in the PR description. Do not invent scope.

## 1. Product
A macOS background app for one user. It reads memories from minimi through its MCP connector and keeps a timeline per person.
- **Card:** when the user returns to a person's native WhatsApp chat, show a quiet corner card with relevant context that is not already on screen. If there is none, show nothing.
- **Ask:** the user can ask about a person, from the card or a global hotkey.

Answers are built at request time. **Never store or display a precomputed summary or headline.** This is general context around a person. It is not an obligations or "what you owe" tracker, and must not rank promises above other context.

Out of scope: Gmail or other return triggers, a thing or project level under a person, vector search, executing tasks, sending messages, teams, reading minimi's database.

## 2. Hard constraints
- Runs on **Python 3.9**. Use `from __future__ import annotations`. No `match`, no runtime `X | Y`, no 3.10+ stdlib.
- Runtime dependencies: stdlib, `pyobjc-core`, `pyobjc-framework-Cocoa`, `pyobjc-framework-ApplicationServices`. No LLM SDK, no HTTP library, no ORM.
- Dev dependencies: `pytest`, `ruff`, `mypy`.
- Everything except `ax_macos.py`, `ui_panel.py` and `app.py` must import and test on Linux with no pyobjc. Import pyobjc only inside those three files.
- No secrets, personal data or machine-specific paths in the repo. All configuration comes from the environment.
- Logs never contain message content, quotes, prompts or responses. Log IDs, counts, durations and error types.
- UI work runs on the main thread. LLM and MCP calls never run on it.
- No bare `except`. A failure in compile, retrieval or the card must log and leave the app running, showing nothing rather than something unverified.

## 3. Layout
```
pyproject.toml
src/person_context/
  config.py      Config dataclass, loaded and validated from env
  models.py      Dataclasses from section 5
  store.py       SQLite, schema migrations, all SQL lives here
  minimi.py      MinimiClient protocol, StdioMinimiClient, call budget
  llm.py         LLMClient protocol, GeminiClient
  prompts.py     All prompt text and JSON response schemas
  privacy.py     Alias gate
  identity.py    Alias proposals and confirmation
  compile.py     Incremental compile job
  retrieval.py   Retriever protocol, WholeTimelineRetriever
  grounding.py   Quote verification and on-screen filter
  screen.py      Pure parser from AX nodes to ScreenContext
  trigger.py     Pure card gate
  service.py     Orchestrates card and Ask, no UI imports
  ax_macos.py    Reads WhatsApp's AX tree into AXNode
  ui_panel.py    Card, Ask and identity confirm panels
  app.py         NSApplication, polling, scheduler, hotkey
  cli.py         track | compile | ask | people | confirm | status | run
tests/
```
Use `typing.Protocol` for `MinimiClient`, `LLMClient`, `Retriever`, with fakes in `tests/`. Expose the CLI as a `person-context` console script.

## 4. Configuration
| Env var | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | required | Never logged |
| `MINIMI_MCP_COMMAND` | required | JSON array, the command that starts the stdio server |
| `MINIMI_MCP_ENV` | `{}` | JSON object merged into that process's environment |
| `PC_DB_PATH` | `~/.person-context/context.db` | |
| `PC_MODEL_COMPILE` | `gemini-3.5-flash-lite` | |
| `PC_MODEL_ANSWER` | `gemini-3.6-flash` | |
| `PC_GAP_HOURS` | `6` | Minimum time away from a person. Accepts decimals |
| `PC_COOLDOWN_HOURS` | `2` | Minimum time between cards for a person. Accepts decimals |
| `PC_COMPILE_HOURS` | `6` | |
| `PC_MCP_WEEKLY_BUDGET` | `20` | `0` means unlimited |

Invalid or missing required values raise a clear error at startup that names the variable.

## 5. Data model
```
Person: id, display_name, created_at
Alias: person_id, value, kind(name|email|phone|company), status(proposed|confirmed|rejected), evidence_quote
Event: id, person_id, occurred_at, source_app, source_thread, speaker(self|them|other|unknown),
       text, source_quote, ingested_at
```
- `Event.id` is the sha256 of `person_id|source_app|normalized source_quote`. Inserts are idempotent.
- Also store: `person_seen(person_id, last_seen_at, last_card_at)`, `compile_state(key, value)` for the watermark, `mcp_calls(called_at, tool)`.
- `normalize(text)`: Unicode NFKC, strip U+200E and U+200F, collapse whitespace, casefold. There is exactly one implementation, used everywhere.
- Use WAL mode and versioned migrations. Each thread uses its own connection.

## 6. Minimi client
- JSON-RPC 2.0 over stdio, newline-delimited: `initialize`, `notifications/initialized`, `tools/call`.
- Start one process per batch and terminate it after. Enforce a 30-second timeout per call. Handle a dead process and non-JSON stdout lines.
- Methods: `search_memory(query, time_range, source_app=None, limit=20, cursor=None)`, `get_latest_context(hours, limit)`, `list_active_threads(window)`.
- `time_range` for look-backs is `{"last": {"hours": n}}`. Use `cursor` for paging, at most 3 pages per query.
- Tool output is markdown whose format can change. **Do not parse its fields.** Split on result boundaries only as far as needed for the alias gate. The real `search_memory` output is a `## Search Results (n)` preamble, then one `### Result <n>` heading per result with `Source`, `Captured`, `Memory` and `Context` lines; the context is raw screen text inside a ``` fence and can contain its own headings and rules. Scan line by line, track ``` fences, and start a block only at a heading matching `Result <number>` outside a fence. Drop the preamble before the first boundary.
- Fail closed: if no boundary is found, return no blocks and log a warning with the character count only. Never treat the whole response as one block.
- Record every `tools/call` in `mcp_calls`. Before a call, if calls in the past 7 days have reached the budget, raise `BudgetExceeded`. Callers catch it and continue without new data.
- A tool result with `isError`, or a quota message, raises `MinimiError`.

## 7. LLM client
- `generateContent` REST through `urllib`, key in the `x-goog-api-key` header. Use `responseMimeType: application/json` and `responseSchema`.
- Timeout 60 seconds. On 429, 500, 503 or timeout, retry up to 3 times with exponential backoff and jitter. Then raise `LLMUnavailable`.
- Validate the parsed response against the expected structure. Invalid output is treated as a failure and retried once.

## 8. Privacy gate
This must run before any text reaches the LLM. Google may review content on the free tier, and the user's other chats must never be sent.
- Only confirmed or proposed aliases of a tracked person can pass.
- A block passes only if it contains an alias on a word boundary after `normalize`.
- Everything else is dropped and counted. Tests must prove that a block about someone else never reaches `LLMClient`.

## 9. Identity
- `person-context track "<name>"` creates a person with a confirmed name alias.
- Compile may propose new aliases through the LLM, such as an email address or company, only with an `evidence_quote` that passes grounding.
- Proposed aliases extend the search but are shown for confirmation. Confirm or reject in one click on the panel, or with `person-context confirm`.
- A rejected alias is never proposed again and no longer passes the gate.
- Never merge two people automatically.

## 10. Compile
For each tracked person:
1. Read the watermark. Default look-back on first run is 14 days.
2. Call `search_memory` for each non-rejected alias, deduplicated. Pass results through the privacy gate.
3. If nothing new passes, stop without calling the LLM and leave the watermark alone.
4. Send the blocks with the `PC_MODEL_COMPILE` model. Extract events for this person with `text`, `source_quote`, `occurred_at`, `source_app`, `source_thread`, `speaker`, plus any alias proposals. Prompt rules: capture all context about the person, including facts, plans, numbers, preferences, people and changes. Do not filter for promises. `source_quote` is copied word for word. Never guess the speaker: use `unknown`.
5. Drop any event whose `source_quote` is not found in the source blocks after `normalize`.
6. Store events in one transaction, then advance the watermark to the start time of this run.

A failure leaves the watermark unchanged, so the next run retries. Running compile twice creates no duplicates. Runs do not overlap: guard with a lock. Triggers are the `PC_COMPILE_HOURS` timer and `person-context compile`.

## 11. Screen parsing
`screen.py` takes a tree of `AXNode(role, identifier, title, value, description, children)` and returns `ScreenContext(person_name, messages)` or `None`. These facts come from native WhatsApp on macOS:

| Element | How to find it | Example |
|---|---|---|
| Open chat name | `AXButton`, identifier `NavigationBar_HeaderViewButton`, title or description | `Bhavya` |
| Message list | identifier `ChatMessagesTableView` | description `Messages in chat with Bhavya` |
| Message | identifier `WAMessageBubbleTableViewCell`, any role | see below |
| Composer | `AXTextArea`, identifier `ChatBar_ComposerTextView` | never read or modify it |

Sample outgoing message description: `Your message, Hi, 10:05 AM, Sent to Bhavya, Red`. Strings can start with U+200E. Treat text beginning `Your message,` as `self`, and any other message as `them`, keeping the full text for the on-screen filter. Skip the end-to-end encryption notice. Unknown shapes must not crash: keep the raw text with speaker `unknown`.

No open chat, or no header found, returns `None`. Build the test fixture from this table.

## 12. Trigger
`trigger.py` is pure: `should_retrieve(person, now, last_seen_at, last_card_at, cfg) -> Decision(bool, reason)`.
- Fire only if the person is tracked with a confirmed alias matching the header, the gap since `last_seen_at` is at least `PC_GAP_HOURS`, and the cooldown has passed.
- No `last_seen_at` counts as a gap. Update `last_seen_at` while the chat stays on screen, including when no card fires.
- At most one retrieval per return. Switching away and back inside the gap does nothing.

`app.py` polls once per second: check the frontmost app cheaply, and read the AX tree only when it is `net.whatsapp.WhatsApp`. Debounce a new chat for 1.5 seconds so the list has loaded.

## 13. Retrieval, card and Ask
`Retriever.retrieve(person_id, question, screen=None) -> list[Event]`. `WholeTimelineRetriever` returns the person's events oldest first, capped at the newest 300. It must stay swappable with no changes elsewhere.

**Freshness:** before building a card or an answer, fetch memories since the last compile through the same privacy gate and extract them as events. If `BudgetExceeded` or `MinimiError` occurs, continue with stored events and log it.

**Card:** the question is what is on screen. The model `PC_MODEL_ANSWER` returns `{show, lines: [{text, event_ids, quote}]}`, at most 4 lines. Prompt rules: only what helps with this conversation right now, nothing already visible, no generic summary, newer facts over older ones with the change stated, and no advice. Grounding then runs:
1. Every line needs a known event ID, and its quote must appear in that event's `source_quote` after `normalize`.
2. Drop a line if its quote appears in the on-screen messages.
3. No lines left, or `show` false: no card, no error, and `last_card_at` is unchanged.

**Ask:** the input is the user's question. It starts on the person on screen. Another person can be named, matched by alias; if ambiguous or unknown, it says so. It returns `{answer, citations: [{event_id, quote}]}`. Citations go through the same check. An answer with no surviving citations displays `I don't have that in this person's context.` Older facts that have since changed must be flagged in the answer. Ask does not save chat history.

## 14. UI
- Non-activating `NSPanel` subclass, bottom-right of the main screen, floating, on all spaces. It must never take focus when the card appears. `canBecomeKeyWindow` returns true so the Ask field can receive typing only after a click or the hotkey.
- Card: person name, lines each with quote, app and date, an Ask field, a dismiss button. No sound. Auto-hide after 45 seconds unless hovered or in use.
- A pending alias shows one line with Confirm and Reject.
- Hotkey: Cmd+Shift+J through a global `NSEvent` monitor. It toggles the Ask panel.
- Run as an accessory app with no Dock icon. Use system colors and fonts so light and dark mode both work. Show "Working" while Ask runs and an inline message on failure.

## 15. Tests and done
Tests use fakes and never need the network, macOS or an API key. Required coverage:
- `normalize` and grounding, including U+200E, and a fabricated quote being dropped
- Privacy gate: another person's block never reaches the LLM
- Compile: idempotent, watermark unchanged on failure and on empty results, budget exceeded, fabricated quote
- MCP client against a fake stdio server script: success, tool error, timeout, dead process, noise on stdout
- LLM client with a mocked `urlopen`: retry then success, retries exhausted, invalid JSON. Assert the key is never in logs
- Screen parser using the section 11 fixture, plus unknown shapes
- Trigger: gap, cooldown, untracked, unconfirmed, repeat return
- Rejected alias behaviour
- `tests/test_scenario.py`, end to end with fakes: a customer's email context exists, the WhatsApp screen shows a vague message, the card shows the off-screen context, visible text is filtered, an unrelated chat fires nothing, and Ask answers with citations and flags a changed number

Done means all of these pass on Python 3.9 if available, otherwise 3.11 with the 3.9 syntax rules kept:
```
pytest -q
ruff check .
mypy --strict src/person_context --exclude 'ax_macos|ui_panel|app'
```
`ax_macos.py`, `ui_panel.py` and `app.py` cannot be tested in CI. Keep them thin and free of business logic, and make them import-safe on macOS. They get reviewed and run on a real Mac afterwards.

Open one PR against `main`. Keep commits small and meaningful, comments only where the reason isn't obvious, and no dead code, TODOs or speculative options. In the PR description, list every deviation from this spec, or state there are none.
