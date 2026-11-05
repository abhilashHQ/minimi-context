# Running the harness demo

Everything below runs from this folder with `.venv/Scripts/python.exe` (shown as `py`).

## What's on each side

- **minimi-today** (left): minimi's 4-tool connector shape (`search_memory`, `list_active_threads`, `get_latest_context`, `meeting_memory`) over one memory per message. That's minimi's memory as it works today, with the Gmail and Calendar connectors feeding it.
- **minimi-harness** (right): the harness. It has 3 tools (`minimi_get_context`, `minimi_search_memory`, `minimi_get_source`) over memories that get updated, ended and merged as messages arrive.
- **Data**: both read the same fictional month (scenario D), with "now" at Thu Nov 26 2026, 8 PM IST. It has 1,276 WhatsApp messages and 50 emails seen on screen. The connectors add 354 more emails: copies of the 50, about 300 promos, and some duplicate deliveries. There are 8 calendar events, one of them stale (Nov 20 dinner still at 8 PM after it moved to 9:30).

## One-time setup

1. Make fresh demo copies of both stores:
   ```
   py -m harness.demo reset
   ```
2. Fully quit Claude Desktop (tray icon > Quit). Then point it at both servers:
   ```
   py -m harness.demo config
   ```
   This backs up your config to `claude_desktop_config.json.bak-before-harness` and replaces `mcpServers` with the two demo servers. Your old `memory` server is in the backup. Run `py -m harness.demo restore` to put it back.
3. Start Claude Desktop. Use one chat per side: in each chat, turn on only one of the two connectors from the tools menu.

## The live moment

Sam moves tomorrow's dinner from 9:30 to 10. This one command adds the two messages to both sides:
```
py -m harness.demo live
```
- **Messages added**: "Sam: Can we push tomorrow's dinner to 10? Class is running late again" and "You: Sure, 10 works", said at 7:55 PM, just before "now".
- **Harness**: it runs one reconcile call. The 9:30 dinner memory is replaced, and the next answer shows "10 PM (was: 9:30 PM, changed by Sam's WhatsApp message)". You don't need to restart the app; the server rereads the store on the next call.
- **Baseline**: it gets one more memory line, the same way it would in minimi today.
- **Timing**: it takes about 10 seconds (two Bedrock calls plus model load).
- **Retakes**: quit Claude Desktop, run `py -m harness.demo reset` (both sides go back to before the live moment), then start the app again. The reset can't overwrite the stores while the app's servers have them open.

## Questions (C1 to C8, same for both sides)

1. What time is dinner with Sam tomorrow?
2. What do I still owe people?
3. Do I owe Priya anything?
4. Did I call the landlord?
5. What did I agree to last week, Monday to Sunday?
6. Does Sam have any food restrictions?
7. What's Sam's flight number?
8. What's on this weekend?

## Scorecard

- **Where it is**: `runs/score_d/scorecard.md`.
- **How it was made**: each question ran 3 times per side through headless `claude -p` (Opus), with only that side's connector enabled. An Opus judge then graded every answer against the required elements in `corpus/scenario_c_questions.yaml`, seeing only the question, the rubric and the answer.
- **Columns**: runs passed, mean tool calls, mean tokens read from memory, and mean time.

To rerun it:
```
py -m runner.run_questions --system harness --db runs/harness/d.db --now 2026-11-26T20:00:00+05:30 --questions corpus/scenario_c_questions.yaml --runs 3 --out runs/score_d/harness
py -m runner.judge --results runs/score_d/harness --questions corpus/scenario_c_questions.yaml
py -m runner.scorecard --system minimi_shape=runs/score_d/minimi_shape --system harness=runs/score_d/harness --out runs/score_d/scorecard.md
```

## Poking at the harness directly

```
py -m harness.cli ask --db runs/demo/harness.db "What do I still owe people?"
py -m harness.cli score --db runs/harness/d.db
```
`ask` prints exactly what `minimi_get_context` hands Claude.

## Rebuilding from scratch

```
py -m corpus.scenario_d build
py -m harness.cli replay --corpus corpus/scenario_d.json --db runs/harness/d.db
py -m baseline.build --scenario corpus/scenario_d.json --db runs/baseline_d/d.db --llm bedrock
```
Both use gpt-oss-120b on Bedrock (us-east-1). The last full harness replay made 936 calls (916 write, the rest dream) in 52 minutes, with 2.0M tokens in and 0.45M out, which comes to under $1. The calls are cached inside the store, so replaying into the same store with the same prompts costs nothing. A fresh store re-calls everything.
