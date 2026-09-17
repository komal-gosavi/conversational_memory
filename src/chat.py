import argparse
from typing import List, Protocol
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from .config import load_settings
from .llm_provider import build_llm
from .memory_buffer import BufferMemoryStrategy
from .memory_hybrid import HybridMemoryStrategy

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use the conversation history "
    "to stay consistent with what the user told you earlier."
    """Core Rules:
    1. Maintain context: Track key facts, user preferences, and ongoing topics from prior turns in the chat history.
    2. Resolve references: Connect ambiguous pronouns (like "it" or "she") to the correct previously mentioned entities.
    3. Adapt and update: If the user corrects or updates a past detail, immediately adopt the new information and discard the old version.
    4. Be concise: Do not repeat full conversation logs back to the user; just apply the contextual memory naturally in your responses."""
)


class MemoryStrategy(Protocol):
    def save_turn(self, human_input: str, ai_output: str) -> None: ...
    def get_context_messages(self) -> List[BaseMessage]: ...
    def get_stats(self) -> dict: ...
    def clear(self) -> None: ...


class ChatSession:
    def __init__(self, llm: BaseLanguageModel, memory: MemoryStrategy) -> None:
        self.llm = llm
        self.memory = memory

    def send(self, user_input: str) -> str:
        history = self.memory.get_context_messages()
        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(history)
        messages.append(HumanMessage(content=user_input))

        response = self.llm.invoke(messages)
        ai_text = response.content

        self.memory.save_turn(user_input, ai_text)
        return ai_text

    def stats(self) -> dict:
        return self.memory.get_stats()


def build_session(memory_kind: str) -> ChatSession:
    settings = load_settings()
    llm = build_llm(settings)

    if memory_kind == "buffer":
        memory: MemoryStrategy = BufferMemoryStrategy()
    elif memory_kind == "hybrid":
        memory = HybridMemoryStrategy(llm=llm, max_token_limit=settings.memory_max_token_limit)
    else:
        raise ValueError(f"Unknown memory kind: {memory_kind!r}")

    return ChatSession(llm=llm, memory=memory)


def _run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", choices=["buffer", "hybrid"], default="hybrid")
    args = parser.parse_args()

    session = build_session(args.memory)
    print(f"[memory={args.memory}] Type 'exit' to quit, 'stats' for memory state.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "stats":
            print(session.stats())
            continue

        reply = session.send(user_input)
        print(f"AI: {reply}\n")


if __name__ == "__main__":
    _run_cli()
