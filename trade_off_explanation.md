# Trade-off Explanation: Buffer vs. Summary vs. Hybrid Memory

This project implements and directly compares two of LangChain's memory
strategies — `ConversationBufferMemory` and `ConversationSummaryBufferMemory`
(the hybrid) — against Groq's free-tier hosted models. A third strategy,
pure `ConversationSummaryMemory`, is discussed below for completeness
(it's a real LangChain option) but wasn't implemented, for reasons
explained in its section.

## 1. Buffer memory (`ConversationBufferMemory`)

Keeps every turn of the conversation verbatim, forever. Nothing is
dropped, nothing is compressed.

**Pros**
- Zero information loss — every word the user ever said is available to
  the model, verbatim.
- Zero extra LLM calls — no summarization step, so it's the cheapest and
  fastest option per turn.
- Simplest to implement, reason about, and debug (`get_stats()` in this
  project always shows `has_summary: False`).

**Cons**
- Token usage grows **linearly and unboundedly** with conversation length.
  In this project's own test run (`test_buffer_memory_does_not_crash_on_long_conversation`),
  36 turns produced 72 stored messages with no ceiling in sight.
- Eventually exceeds the model's context window on a long enough
  conversation — at that point requests fail outright, or (on models that
  silently truncate) old context is lost anyway, just without the
  developer controlling *which* part is lost.
- Cost scales with conversation length even on providers with per-token
  pricing, since the full history is resent on every single turn.

**When to use it**
- Short-lived sessions (a support ticket, a single Q&A exchange, a
  scripted demo) that are very unlikely to approach the model's context
  window.
- Debugging and evaluation, where you want to *see* the raw, unaltered
  conversation history rather than an LLM's paraphrase of it.
- Any situation where losing or distorting a single word of context is
  unacceptable (e.g., legal or medical transcripts) and the conversation
  length is bounded by design.

## 2. Pure summary memory (`ConversationSummaryMemory`)

Compresses *every* turn into a running summary immediately — there is no
verbatim window at all. Not implemented in this project, but worth
contrasting directly since it's the other end of the spectrum from
buffer memory.

**Pros**
- Token usage stays flat regardless of conversation length — the
  strongest guarantee against context-window overflow of the three
  options.
- Cheapest *sustained* token cost on long conversations, since the
  historical context sent on each turn is always just one short summary
  paragraph.

**Cons**
- An LLM call to update the summary fires on **every single turn**, not
  just when a threshold is crossed — the highest number of extra LLM
  calls of the three strategies, which matters a lot on a rate-limited
  free tier like Groq's.
- Even the most recent turn is immediately paraphrased rather than kept
  verbatim, so exact wording, phrasing, and fine-grained detail from the
  last thing the user said can be softened or dropped a turn later.
- Summary quality is entirely dependent on the summarizing LLM; a weak
  or fast model can compound small errors turn over turn since each new
  summary is built from the previous (possibly already-lossy) one.

**When to use it**
- Very long-running conversations (assistants that persist over days or
  weeks) where flat token cost matters more than perfect recall of
  exact recent wording.
- Use cases that only need the *gist* of history (e.g., "has this user
  mentioned being a returning customer?") rather than precise quotes.

## 3. Hybrid summary-buffer memory (`ConversationSummaryBufferMemory`) — implemented here

Keeps recent turns verbatim; once the buffered history's token count
crosses `MEMORY_MAX_TOKEN_LIMIT`, the oldest turns are popped off and
folded into a running summary via one extra LLM call. This project's
`memory_hybrid.py` implements this with a custom tiktoken-based token
counter (see `README.md` → Design decisions) so it works with Groq
models, which don't ship the tokenizer LangChain expects by default.

**Pros**
- Bounded token growth like pure summary memory, but the *most recent*
  turns stay verbatim — so precise recall of the last few exchanges is
  as good as buffer memory.
- The extra LLM call for summarization only fires when the threshold is
  actually crossed, not on every turn — cheaper than pure summary memory
  in practice, especially on shorter conversations.
- This project's test suite (`tests/test_conversation_memory.py`) proves
  this concretely: with `max_token_limit=300`, a secret detail
  ("my dog's name is Nimbus") stated in turn 1 was still correctly
  recalled in turn 36, *after* `get_stats()["has_summary"]` confirmed
  summarization had actually triggered — i.e., the detail survived being
  folded into a summary and paraphrased back out correctly.

**Cons**
- More moving parts than either pure strategy: a token counter, a prune
  threshold, and a summarization prompt all have to work correctly
  together. This project's own tiktoken-fallback logic (network-safe
  approximate counting) is a direct consequence of that extra
  complexity.
- Still not lossless — once a turn is folded into the summary, exact
  wording from it is gone, same risk as pure summary memory, just
  delayed until the threshold is crossed instead of immediate.
- The summarization LLM call is a real, occasional latency spike:
  most turns are as fast as buffer memory, but the turn that crosses the
  threshold takes noticeably longer (one extra round-trip to the model).
- Requires picking a reasonable `max_token_limit` — set it too low and
  you pay the summarization cost almost every turn (approaching pure
  summary memory's cost); set it too high and you approach buffer
  memory's context-overflow risk before summarization ever kicks in.

**When to use it**
- The general-purpose default for most conversational agents: long
  enough sessions that buffer memory's unbounded growth is a real risk,
  but where recent-turn fidelity still matters (the assistant needs to
  remember exactly what the user just asked, not a paraphrase of it).
- Customer support, coding assistants, and tutoring bots — anywhere a
  user might reference something specific from a few turns ago and
  expect exact recall, while the conversation as a whole could run long.
- This project's actual constraint: a free-tier, rate-limited LLM (Groq)
  where every LLM call has a cost in both money-adjacent quota and
  request-per-minute budget, so it's worth avoiding pure summary
  memory's per-turn summarization call, but a real risk of long test/demo
  conversations makes buffer memory's unbounded growth unsafe too.

## Summary table

| | Buffer | Pure summary | Hybrid (implemented) |
|---|---|---|---|
| Token growth | Unbounded | Flat | Bounded |
| Recent-turn fidelity | Perfect | Paraphrased immediately | Perfect (until threshold) |
| Old-turn fidelity | Perfect | Paraphrased | Paraphrased (after threshold) |
| Extra LLM calls | None | Every turn | Only when threshold crossed |
| Risk of context-window overflow | High on long chats | None | Low (tunable via threshold) |
| Implementation complexity | Lowest | Low | Highest |
| Best for | Short, bounded sessions | Very long sessions, gist-only recall | General-purpose long sessions needing recent-turn accuracy |
