# L1-04: Conversational Memory Management

> A practical comparison of buffer and hybrid summary-buffer memory strategies for LLM conversations using LangChain and Groq.

## Overview

LLMs are stateless, so applications need to provide conversation history with each request.

This project implements and compares two memory strategies:

- **Buffer Memory** — keeps the complete conversation history.
- **Hybrid Summary Buffer** — keeps recent messages and summarizes older messages when the token limit is reached.

## Features

- Buffer and hybrid conversation memory
- Token-aware history management using `tiktoken`
- Groq LLM integration
- Configurable model and memory token limit
- Interactive CLI
- Tests for long conversations and memory recall

## Tech Stack

- **Python**
- **LangChain**
- **Groq**
- **tiktoken**
- **python-dotenv**
- **pytest**

## Project Structure

```text
memory_manager/
├── src/
│   ├── config.py
│   ├── llm_provider.py
│   ├── memory_buffer.py
│   ├── memory_hybrid.py
│   └── chat.py
├── tests/
│   └── test_conversation_memory.py
├── examples/
├── requirements.txt
└── README.md
```

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

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd conversational_memory
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env`:

```env
GROQ_API_KEY=your_api_key
GROQ_MODEL=openai/gpt-oss-20b
MEMORY_MAX_TOKEN_LIMIT=1000
```

## Usage

Run with hybrid memory:

```bash
python -m src.chat --memory hybrid
```

Run with buffer memory:

```bash
python -m src.chat --memory buffer
```

Inside the CLI:

```text
stats   # Show memory state
exit    # Exit the application
```

## Testing

```bash
pytest tests/ -v
```

The tests verify that hybrid memory summarizes older conversation history while retaining important information.

## Roadmap

- Migrate to LangGraph checkpointer-based memory
- Add persistent storage
- Add retry/backoff for API failures
- Support multiple conversation sessions

## Contributing

Contributions are welcome. Please open an issue or pull request for improvements.

## License

No license has been specified yet.