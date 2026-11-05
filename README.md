# minimi memory harness

A memory layer that sits between minimi's capture and the model that reads it. It keeps memory small and current as connectors multiply, so one cheap tool call answers the question instead of five.

minimi just added Gmail and Calendar connectors. Now the same fact arrives from three places (the WhatsApp chat on screen, the email, the calendar invite), some copies are stale, and a flood of promo mail rides along. On an append-only pile of memories that is context rot: more near-duplicates that look like the answer. The harness decides, at write time, what each new message changes.

<p align="center"><img src="docs/img/headline.svg" alt="Harness 22 of 24 answers correct vs minimi today 14 of 24; 1.7 vs 4.9 tool calls; 1,085 vs 4,319 tokens read from memory"></p>

Same fictional month on both sides, same reader (Claude Opus), same 8 questions, 3 runs each. Details and caveats in [How it was scored](#how-it-was-scored).

## See it

**"What time is dinner with Sam tomorrow?"** Sam moved last week's dinner from 8 to 9:30 by email, then asked on WhatsApp "Dinner Friday, same as last time?". minimi today finds the old 8 PM calendar entries and hedges. The harness answers in one call and shows where 9:30 came from.

![C1 side by side](docs/img/c1_dinner.png)

**What Claude actually reads for "What's on this weekend?"** Six of minimi's top ten hits are promo mail from the new Gmail connector, plus a brunch from three weeks ago. The harness hands over the whole answer in 159 tokens.

![C8 what Claude reads](docs/img/c8_what_claude_reads.png)

**A new message arrives.** Sam pushes dinner to 10. minimi appends two more lines next to the 9:30 ones. The harness replaces the memory, keeps 9:30 as history, and says what changed it.

![Live moment](docs/img/live_moment.png)

## Every question, every run

<p align="center"><img src="docs/img/per_question.svg" alt="Per-question results: three runs each for minimi today and the harness, with mean tool calls"></p>

| | minimi today | harness |
|---|---|---|
| Runs judged correct | 14 / 24 | **22 / 24** |
| Questions correct on all 3 runs | 4 / 8 | **6 / 8** |
| Tool calls per question | 4.9 | **1.7** |
| Tokens read from memory per question | 4,319 | **1,085** |
| Answer time | 40.1 s | **33.6 s** |

Per-question numbers: [SCORECARD.md](SCORECARD.md).

## How it works

Four jobs, borrowed from the memory systems that already work and fitted to minimi's shape (screen capture plus connectors in, local SQLite out, MCP to whatever model the user picks).

| Job | What happens | Where it comes from |
|---|---|---|
| **Write** | Raw messages are stored and deduplicated across capture paths. Promo and bulk mail is gated out. Messages are grouped into bursts per thread (60 min quiet, 4 h, or 20 messages). One LLM call per burst sees the burst plus up to 40 related memories (same thread, same people, similar text) and returns ops: `new`, `same`, `replaces`, `extends`, `ends`. Every op must quote its message word for word; code checks the quote, the target, the people and the time order before applying it. | Supermemory (grouping, updates/extends, latest flag), Mem0 (candidates in one reconcile call), Zep (said-at vs seen-at, invalidate instead of delete) |
| **Retrieve** | `minimi_get_context(question)` with no LLM at read time: parse the question into filters (owe, waiting on, this weekend, last week...), fuse vector, BM25 and filter lists with RRF, weight by time fit, drop near-duplicates, attach one hop of history ("was 8 PM, changed by Sam's email"), add a Changed recently section, cap at 1,500 tokens, and say in a footer if anything was cut. | Zep (RRF + MMR), Generative Agents (recency), Chroma's context rot report (never show a stale version as a peer) |
| **Forget** | Nothing is deleted. Replaced and ended memories stay as history; statuses expire; past events and old chatter decay in ranking only. | Mem0 v3 (never destroy), Supermemory (decay by kind) |
| **Dream** | A background pass at night: expire, retry failed bursts, settle contradictions (same-day events, related facts about one person), close stale commitments against later evidence ("Thanks for calling" closes "call the landlord"), merge near-duplicates. | Letta sleep-time agents |

Full design: [docs/design.md](docs/design.md). Code: [`harness/`](harness).

```
capture + connectors ──> W1 store raw ──> W2 gate ──> W3 bursts ──> W5 candidates ──> W6 one LLM call
                                                                                         │
 Claude ── minimi_get_context ──> R1..R10 (no LLM) ──> SQLite memories <── W7 validate + W8 apply
                                                            ▲
                                                 dream (nightly): expire, retry, settle, close, merge
```

What the final store holds after a month of 1,334 messages and 354 connector emails: **61 lasting memories** (5 open todos, 11 events, 14 facts, 31 preferences) plus a log of chat episodes that decay in ranking. All 19 end-state checks pass: exactly the five real open todos, the four done ones closed (one of them only through the dream), the stale 8 PM invite never revived, Priya's flight number kept on Priya.

## How it was scored

- **Data is fictional.** `corpus/scenario_d.json`: a month of one person's WhatsApp (seen on screen) and Gmail, plus what Gmail and Calendar connectors would add: copies of every email, 300 promo mails with deliberate distractors ("Friday dinner for two at 8 PM", "6E 2131 to Delhi"), 8 calendar events including one stale invite.
- **"minimi today" is a clone of the shape, not minimi.** `baseline/minimi_shape.py` exposes the four tools of minimi's hosted connector (`search_memory`, `list_active_threads`, `get_latest_context`, `meeting_memory`) over one memory per message, written by the same model the harness uses (gpt-oss-120b). It's how minimi's memory works today as far as the public connector shows; it is not minimi's code.
- **Reader:** headless `claude -p` with Opus and only one side's connector. **Judge:** Opus seeing only the question, the rubric in `corpus/scenario_c_questions.yaml` and the final answer, never the tool calls. I read every failed verdict; the harness's two fails are date precision (a "this weekend" deadline given as Saturday; a sent date not stated) and stay counted as fails.
- **Cost:** the harness replay of the whole month is 942 gpt-oss-120b calls on Bedrock, about an hour, under $1. Reads cost nothing.
- **Known gaps:** chat episodes are still stored (they decay, they don't vanish); Bedrock is not deterministic at temperature 0, so a fresh replay can land a check or two differently; the corpus is one person and one month.

## Run it

```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e .
python -m corpus.scenario_d build
python -m harness.cli replay --corpus corpus/scenario_d.json --db runs/harness/d.db   # needs AWS Bedrock access
python -m harness.cli ask --db runs/harness/d.db "What do I still owe people?"
python -m harness.server --db runs/harness/d.db                                         # MCP over stdio
```

Demo with both sides in Claude Desktop, the live moment, and the full scorecard: [DEMO.md](DEMO.md). The images in this README are rebuilt from the run data by `python -m docs.make_assets`.

Earlier work in this repo (the P1 search-only baseline): [docs/p1-baseline.md](docs/p1-baseline.md).
