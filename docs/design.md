> Design as settled on 2026-11-04. Two things changed in the build: the write model is gpt-oss-120b on Bedrock (not Gemini flash-lite), and replay feeds messages in arrival order (when minimi saw them), which is how live connectors deliver.

# minimi memory harness: final design

2026-11-04. The doc we build from. Four jobs: write, retrieve, forget, dream. It runs on minimi's shape: Accessibility capture plus Gmail and Calendar in, memory sentences, local SQLite and vectors, MCP out.

## The problem

1. minimi's memory is an append-only pile of sentences searched by vector similarity.
2. As the pile grows, plain search breaks. It returned dinner at 8 PM and at 9:30 PM as both current, gave a promo email for "what should I do first tomorrow", and kept duplicates of one page.
3. On 1,326 fictional messages, a strong model with plain search needed 6 to 12 tool calls per question and never listed everything the user owed (0 of 3 runs).
4. Gmail and Calendar connectors multiply the volume and send the same fact from several places, some copies stale. That is context rot: more near-duplicates that look like the answer.
5. minimi doesn't choose the reader model, and the free plan allows 20 connector calls a week, so one cheap call has to answer.

## What we took from whom

Grouping before extraction, the updates and extends relations, a latest flag with history kept, and decay by kind come from [Supermemory](https://supermemory.ai/docs/concepts/graph-memory) ([formation](https://supermemory.ai/docs/concepts/how-it-works), [search](https://supermemory.ai/docs/v5/api-reference/recall-search/search-memories)). Similar memories as candidates in one reconcile decision come from [Mem0's paper](https://arxiv.org/html/2504.19413v1). Single pass and never destroying come from [Mem0 v3](https://docs.mem0.ai/migration/platform-v2-to-v3). We still mark old versions as ended, because Mem0 v3 hands readers about 7k tokens and our budget is 1.5k. From [Zep](https://arxiv.org/html/2501.13956v1): when a thing was said versus when it was seen, invalidate instead of delete, raw episodes linked to facts, and RRF plus MMR. The background dream job is [Letta's](https://docs.letta.com/guides/agents/sleep-time-agents) kind, backed by the [sleep-time compute](https://arxiv.org/html/2504.13171v1) results. Processing a thread once it goes quiet comes from [LangMem](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/). Recency in ranking comes from [Generative Agents](https://arxiv.org/html/2304.03442v2). The small budget and "never show a stale version as a peer" come from [Chroma's context rot report](https://www.trychroma.com/research/context-rot): one similar-but-wrong distractor lowers accuracy.

## Data model (SQLite)

| Table | Fields |
|---|---|
| sources | id, kind (screen, gmail, calendar), external_id, thread_key, author, participants, said_at, seen_at, text, content_hash (unique), skip_reason, burst_id |
| bursts | id, thread_key, first_said_at, last_said_at, status (open, done, failed), attempts |
| memories | id, text, kind, owner (who has to act, for kinds that have one: you or other), people, event_at (time it happens or is due), event_end, expires_at, valid_from, valid_to, end_reason (replaced, ended, expired, merged), source_count, embedding |
| memory_sources | memory_id, source_id, quote, role (created, confirmed, changed) |
| links | from_id, to_id, type (replaces, extends, ends, merged_into), made_by (write, dream), reversed_at |
| ops_log | id, at, burst_id, op, target_id, outcome (applied, downgraded, rejected), reason, raw_json |
| meta | mode (live, replay), now |
| llm_cache | prompt_hash, response |

- **Kinds:** event, todo, fact, preference, status, episode.
- **Open-state kinds:** todo, event and status, which stay open until done, cancelled or expired.
- **Current:** valid_to is null and not expired.
- **Indexes:** FTS5 with porter stemming on memories and sources. Vectors from `all-MiniLM-L6-v2`, held in numpy.
- Nothing is deleted.

## Write

| Box | Step |
|---|---|
| W1 Store raw | Save as a source. Known hash: stop. Screen: keep only lines new since the window's last capture. |
| W2 Gate | Gmail promo, social, or bulk mail from someone you never replied to: stored, but makes no memory. Stop. |
| W3 Form burst | Add the source to its thread's open burst. The burst closes after **60 minutes of quiet, 4 hours after its first message, or at 20 messages**. The rule is the same in live and replay. |
| W4 Calendar shortcut | A version becomes a templated memory, a cancellation ends the event's memories. No LLM unless W5 finds candidates. |
| W5 Candidates | Up to 40, in order. **Thread:** memories from this thread_key, including ones ended in the last 14 days (up to 10). **People:** open-state memories that share a person with the burst (name prefix match, excluding you), plus later-dated memories about those people (up to 15). **Per message:** the top 3 by vector and top 3 by BM25 for each message, which fill the rest. |
| W6 Reconcile call | One flash-lite call per burst. Input: messages b1 to b20 with sender and said_at, the 3 messages before them as context, and candidates c1 to c40. Output: ops, each citing one message and a verbatim quote (table below). Dates resolve against the cited message. A proposal isn't an agreement. The output is the burst's end state. |
| W7 Validate | **Quote:** fuzzy partial match of 90 or more, or reject. **Target:** must be in the candidate list. **Time:** `replaces` and `ends` need a cited said_at later than the target's valid_from, and `closed_by` needs a closer that is later than the cited message; a failing `replaces` is inserted already ended, a failing `ends` is rejected. **People:** a `same` or `replaces` that shares no person with its target is downgraded to `new`, and the pair is logged for the dream. |
| W8 Apply | Apply in said_at order, embed, index, log, close the burst. Rejected ops queue the burst for D2. |

| Op | Meaning |
|---|---|
| new | A memory that nothing covers. Optional `closed_by cX` when a candidate already shows it done or untrue: it is inserted already ended. |
| same cX | Repeats cX. A stale copy confirms the old version and never revives it. |
| replaces cX | cX is no longer true. Gives the new version. |
| extends cX | Adds detail. Both stay true. |
| ends cX | cX is done, cancelled, or stopped being true. |

The write path never reads the clock. Arrival order never decides what is current.

## Retrieve

Three read-only, idempotent MCP tools, markdown by default, per mcp-builder. Server instructions: call `minimi_get_context` first with the question as written; use `as_of` as now; "(was ...)" is history; if the footer shows more than the pack, make the call it names.

| Tool | Inputs | Returns |
|---|---|---|
| `minimi_get_context` | question, time_range, people, max_tokens (default 1,500, range 500 to 4,000) | Pack, "Changed recently", footer |
| `minimi_search_memory` | query (empty means filter only), kind, owner, people, after, before, include_history, limit (25, max 50), cursor | Matches. Filter-only results are sorted by date, with total_count, has_more and next_cursor. |
| `minimi_get_source` | memory_ids (1 to 10) | Quotes, plus each source message with 2 neighbors on each side |

`minimi_get_context`, with no LLM:

| Box | Step |
|---|---|
| R1 Parse question | Turn the question into filters, using the table below. Time words are read against `meta.now`. Names that match stored people become a people filter. |
| R2 Candidates | From current memories: top 40 by vector, top 40 by BM25, plus one list per parsed filter (sorted by event_at, undated last). |
| R3 Fuse | RRF with k = 60 over the lists, scaled to 0 to 1. |
| R4 Time fit | Multiply in. Window: 1.0 inside, 0.3 outside. No window: open items overdue or due within 48 h 1.0, falling to 0.5 at 14 days; past events 0.2; facts and preferences 1.0; episodes halve every 30 days, undated todos every 14. |
| R5 Dedupe | MMR: skip anything above 0.9 cosine to a line already picked. |
| R6 Annotate | Mark a possible conflict when current events on one day share a person, or when the pair is in the downgrade log. Add one hop of history, for example "(was: 8:00 PM, changed Wed by Sam's email)". |
| R7 Changed recently | Run R2 to R5 over memories replaced or ended in the last 14 days. Show up to 5 whose replacement isn't already in the pack, each with when it changed, how, and the quote. |
| R8 Raw fallback | Up to 3 FTS matches from sources not processed yet, marked raw. |
| R9 Pack | Fill lines up to max_tokens. Format: `[id] kind (owner) · time or due · text · "quote" · sources`, with "OVERDUE since ..." where it applies. |
| R10 Footer | For each parsed filter, how many are shown out of the total, then the exact `minimi_search_memory` call for anything that was cut. |

**R1 question-to-filter parse** (data, not code paths; extend by adding rows):

| Words | Filter |
|---|---|
| owe, need to, have to, promised, should I do | kind todo, owner you |
| owes me, waiting on, waiting for | kind todo, owner other |
| pending, outstanding, open | open-state kinds |
| what's on, plans, schedule, free, busy | kind event |
| like, prefer, favorite, allergic, eat, vegetarian | kinds preference, fact |
| tomorrow, this weekend, Friday, next week, last month | time window |

## Forget

| Box | Rule |
|---|---|
| F1 Replaced | A later statement replaces this memory. It is shown only as "(was ...)" and in Changed recently. |
| F2 Ended | An `ends` op, a `closed_by`, or a cancelled event. |
| F3 Expired | A status passes expires_at, which defaults to 7 days after said_at. Checked at read time and set by the dream. |
| F4 Past | An event that has happened stays stored and drops to time fit 0.2, unless the question's window covers it. |
| F5 Decay | Ranking only. Episodes and undated todos lose weight with age. Facts and preferences don't. |
| F6 Noise | Gated mail never becomes memory. |

Ended memories stay reachable through Changed recently for 14 days, and through `include_history` after that.

## Dream

Letta-style background pass. Live: nightly when idle, or after 300 new memories. Replay: once after ingest, at the corpus's `now`. Every step writes through W7 and W8.

| Box | Step | Calls |
|---|---|---|
| D1 Expire | End statuses past expires_at | 0 |
| D2 Retry | Resend failed bursts with candidates recomputed from the current state, plus a note naming the failed ops. A second failure is counted in the report. | 1 per burst |
| D3 Settle | Downgrade-log pairs and same-day events that share a person: one call per group decides replaces, same, or both stand | 1 per group |
| D4 Close stale open items | For each open-state memory, take the top 3 by vector and BM25 among later memories from any thread. Batches of 10 ask whether this evidence closes it. | 1 per 10 items |
| D5 Merge | Unlinked pairs above 0.92 cosine: same, replaces, extends, or unrelated | 1 per 10 pairs |

## Replay

`meta.now` is the only clock. Conversations are fed in the corpus's listed ingest order (scenario C: Gmail, then WhatsApp), with messages in said_at order inside each. Bursts use W3's rule applied to said_at. After the last burst, set `meta.now` to the corpus's `now` (2026-11-26 20:00 IST), run the dream once, and ask the questions. The report lists bursts, calls, cache hits, ops applied, downgraded and rejected (by message), and dream changes. The cache makes reruns free.

## Three user flows

### (a) Dinner moves across apps

**User side** (imagined dates). On Mon Nov 9 Sam texts "Dinner Friday at 8?", and you reply "Yes, 8 works" and add it to Calendar. On Wed Nov 11 Sam emails "Can we do 9:30 on Friday instead of 8?", and you answer "9:30 works. See you then." You open the same WhatsApp chat twice. On Thursday evening you ask Claude "What time is dinner with Sam?"

**Harness.**
- The Gmail burst arrives first: `new` "Dinner with Sam, Fri Nov 13, 9:30 PM".
- The Monday WhatsApp burst arrives later and finds it through people and per-message candidates. W7 sees Monday is before Wednesday and inserts the 8 PM version already ended, replaced by 9:30.
- The calendar event, still at 8 PM, gets `same` on the old version, so it can't revive 8 PM. The second chat capture stops at W1.
- One `minimi_get_context` call returns:

```text
as_of: Thu Nov 12 2026, 8:00 PM IST (Asia/Kolkata)
[m4] event · Fri Nov 13, 9:30 PM · Dinner with Sam (was: 8:00 PM, changed Wed Nov 11 by Sam's email) · "9:30 works. See you then." · gmail
```

### (b) "Do I owe Priya anything?"

**User side.** On Nov 10 Priya emails for the deck, and you promise it by 6 PM Friday. On Nov 13 her mail is down, so you send it on WhatsApp: "Here is the deck." On Nov 20 she asks by email for the cap table, and you promise it by Nov 30.

**Harness.**
- Gmail bursts create todo t1 (Nov 10) and todo t2 (Nov 20). Nothing in the Nov 20 burst says the deck was sent, so t1 stays open.
- The Nov 13 WhatsApp burst pulls t1 and t2 through the people rule ("Priya" matches "Priya Nair"), and pulls t1 again per message. On raw text, "Here is the deck." ranks Priya's request 3rd of 1,325. The whole burst embedded as one ranked it 1,029th.
- The op is `ends t1`, quoting "Here is the deck." It passes the time check (Nov 13 is after Nov 10). An `ends` on t2 would fail, since Nov 13 is before Nov 20.
- R1 reads "owe" as kind todo, owner you, and "Priya" as a people filter. One call returns:

```text
[t2] todo (you) · due Mon Nov 30 · You'll send Priya the cap table · "Will do, I'll send the cap table by November 30." · gmail
Changed recently:
[t1] todo (you) · ended Fri Nov 13 on WhatsApp · Send Priya the deck by Fri Nov 13, 6 PM · "Here is the deck."
Footer: your open todos with Priya 1 of 1
```

The same Changed recently code shows a cancelled dinner, or "Sam eats fish again" replacing "Sam is vegetarian".

### (c) A flooded week, then "what's on this weekend?"

**User side.** Gmail syncs 400 emails in a week: newsletters, sale mails, a calendar invite for Mom's Sunday lunch, and the same invite opened in the browser. On Thursday you ask "What's on this weekend?"

**Harness.**
- W2 drops the promos and bulk mail, so none of it becomes memory.
- The invite email and the calendar event describe one lunch. Whichever arrives second finds the first through the people rule ("Mom") and gets `same`. The browser copy stops at W1, or it's another `same`.
- R1 reads "what's on" as kind event and "this weekend" as Sat Nov 28 to Sun Nov 29. The window list holds every event in that range, and R5 drops leftover near-duplicates.
- The pack holds the weekend's events, with a footer saying "events this weekend 3 of 3". That's about 150 tokens, whether the store has 2,000 memories or 200,000.

## Calls and quota

| Item | Number |
|---|---|
| Write calls, scenario C | 905 bursts (60 min quiet, 4 h, cap 20) |
| Dream calls, scenario C | about 30 |
| Read calls | 0 |
| Tokens per write call | about 3,300 in (instructions 800, burst 800, context 120, 40 candidates 1,600) and 300 out |
| Total, scenario C | about 3.1M in and 0.3M out. Under a dollar at flash-lite-class prices; the `gemini-3.5-flash-lite` price is unverified. |
| Time | about an hour at 15 requests a minute |

**Check the key's quota before building.** On Nov 1 the free tier stopped at 20 requests a day for one model. At that cap, about 935 calls can't run, so use a paid key. A 120-minute quiet gap cuts writes to 506 but doubles live lag. The corpus noise has a median gap of 84 minutes and no bursts, so real chats batch better than this.

## Failure modes

| Failure | Handled by |
|---|---|
| Both versions current | `replaces` (W6), dream settle (D3), conflict mark (R6) |
| Stale copy, out-of-order arrival | `same` on the old version; said_at decides, never seen_at |
| Duplicates | Hash and screen diff (W1), `same`, merge (D5), MMR (R5) |
| Short replies ("Yes, see you then") | Same burst as what they answer, plus 3 messages of context |
| Vague questions | Gate (W2), filter parse (R1), time fit (R4) |
| "List everything" | Filter lists in R2, footer totals (R10) |
| Closure in another thread, or processed first | People and per-message candidates, `closed_by`, D4 |
| Nickname mismatch ("Mom" vs her name) | Downgrade to `new`, conflict mark, D3 |
| Wrong `replaces` or `ends` | Time check, people check, visible in "(was ...)" or Changed recently, one-step undo through links |
| Reader trusts its own clock | `as_of` from meta.now, absolute dates |
| Live processing lag (up to 60 min after quiet) | Raw fallback (R8) |

## LATER

| Item | Why it waits |
|---|---|
| Importance score, access refresh | Matter only when a window overflows the budget; the footer covers that |
| Derived memories, reflection | No question found that a pack of the parents misses |
| Episode rollups, user hard-delete | Volume and privacy features, not needed for the demo |
| Email-address alias table, entity graph | Only if the eval shows person misses |
| Read-time LLM rerank, query rewrite | Cost and latency, and minimi doesn't pick the reader |
| Capture attribution checks | Who-said-what errors from minimi's capture flow into owner and people; fictional replay can't show them |
