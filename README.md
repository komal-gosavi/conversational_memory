# L1-04: Conversational Memory Management

An AI Engineer technical evaluation deliverable implementing and comparing two
LangChain memory strategies — plain buffer memory and a hybrid
summary-buffer memory — against Groq's free-tier hosted LLMs.

## Architecture

```
.env ──► config.py ──► llm_provider.py ──► ChatGroq
                              │
                              ▼
                         chat.py (ChatSession)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
           memory_buffer.py     memory_hybrid.py
           BufferMemoryStrategy HybridMemoryStrategy
```

- **`config.py`** loads and validates environment variables (`.env`) into a
  single immutable `Settings` object. Everything else reads from `Settings`,
  nothing else touches `os.environ` directly.
- **`llm_provider.py`** builds a `ChatGroq` instance from `Settings`. This is
  the one place that knows about the Groq SDK; swapping providers later only
  means editing this file.
- **`memory_buffer.py`** wraps LangChain's `ConversationBufferMemory`:
  keeps the entire conversation verbatim, forever.
- **`memory_hybrid.py`** wraps LangChain's `ConversationSummaryBufferMemory`:
  keeps recent turns verbatim, and once the buffered history crosses
  `MEMORY_MAX_TOKEN_LIMIT` tokens, the oldest turns are popped off and
  folded into a running natural-language summary via an extra LLM call.
- **`chat.py`** ties an LLM and a memory strategy together into a
  `ChatSession`. Both memory classes satisfy the same structural
  `MemoryStrategy` interface (`save_turn`, `get_context_messages`,
  `get_stats`, `clear`), so `ChatSession` never has to know which one it's
  holding — that's what makes the buffer-vs-hybrid comparison
  apples-to-apples: everything else in the pipeline is identical.
- **`tests/test_conversation_memory.py`** drives a 36-turn conversation
  through both strategies against the real Groq API, proving the hybrid
  strategy actually summarizes (and still recalls an early detail
  afterward) and that the buffer strategy survives the same length without
  crashing.

## Setup

```bash
# 1. Enter the project
cd l1-04-memory

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_... (free key: console.groq.com/keys)
```

## Usage

```bash
# Interactive chat demo
python -m src.chat --memory hybrid     # or: --memory buffer

# Inside the chat: type 'stats' to inspect memory state, 'exit' to quit

# Run the memory tests (real Groq API calls, ~2 min)
pytest tests/ -v
```

## Design decisions

**Why `ConversationBufferMemory` / `ConversationSummaryBufferMemory`
despite the deprecation warning.** Both classes have been deprecated since
LangChain 0.3.1 in favor of LangGraph-based persistence (checkpointers),
but they're not scheduled for removal until a future 1.0/2.0 release and
work fully on the `langchain>=0.3.0,<0.4.0` pin used here. They're used
directly because this assignment explicitly calls for a
`ConversationSummaryBufferMemory`-based hybrid strategy compared against
`ConversationBufferMemory`. A production migration path would move to
LangGraph's checkpointer-based memory instead.

**Why a custom tiktoken-based token counter.**
`ConversationSummaryBufferMemory` decides when to summarize by calling
`self.llm.get_num_tokens_from_messages(...)`. OpenAI chat models override
that with `tiktoken`; Groq's `ChatGroq` doesn't, so it silently falls back
to a `transformers`-based tokenizer that isn't installed and raises an
`ImportError`. `memory_hybrid.py` overrides `prune()` to count tokens with
`tiktoken`'s `cl100k_base` encoding instead — not an exact match for
Llama/GPT-OSS's real tokenizer, but consistent turn-to-turn, which is all a
threshold heuristic needs. It's also wrapped in a fallback: if `tiktoken`
can't reach the network to download its encoding file (offline machines,
restricted networks), it falls back to a `len(text) // 4` heuristic rather
than crash — matching the assignment's "must not break on long
conversations" requirement even when the token counter itself is degraded.

**Why Groq-only.** The project originally supported a local Ollama backup
provider behind the same `MemoryStrategy`/LLM interface, but that's been
removed to match actual usage — this project only runs against Groq.

**Why the model is read from `.env` rather than hardcoded.** Groq's
free-tier model catalog has churned through several deprecations in 2026
(`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` were both shut down
for free/developer tiers on 2026-08-16). Keeping `GROQ_MODEL` in `.env`
means a model swap is a one-line config change, not a code change.

**Why the test script sleeps between turns.** Groq's free tier rate-limits
at 30 requests/minute. A 36-turn test makes 36+ calls; a 1-second sleep
after each `send()` keeps the suite comfortably under that limit without
meaningfully slowing the test down.

## Project structure

```
l1-04-memory/
├── .env.example
├── requirements.txt
├── README.md
├── trade_off_explanation.md
├── src/
│   ├── config.py
│   ├── llm_provider.py
│   ├── memory_buffer.py
│   ├── memory_hybrid.py
│   └── chat.py
├── tests/
│   └── test_conversation_memory.py
├── examples/
│   └── sample_run_output.txt
└── docs/
```
