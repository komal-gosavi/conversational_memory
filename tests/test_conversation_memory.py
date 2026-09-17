import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.chat import ChatSession
from src.config import load_settings
from src.llm_provider import build_llm
from src.memory_buffer import BufferMemoryStrategy
from src.memory_hybrid import HybridMemoryStrategy

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)

SECRET_DETAIL = "My dog's name is Nimbus and I live in Austin."
RECALL_QUESTION = "What is my dog's name?"

FILLER_TOPICS = [
    "Tell me a fact about the ocean.",
    "What's a good recipe for pancakes?",
    "Explain how rainbows form.",
    "What's the tallest mountain in Africa?",
    "Give me a tip for better sleep.",
    "What's an interesting fact about octopuses?",
    "How does a refrigerator work?",
    "What's the history of the printing press?",
    "Suggest a beginner workout routine.",
    "What causes seasons to change?",
]


def _send(session: ChatSession, text: str) -> str:
    reply = session.send(text)
    time.sleep(1)  # stay under Groq free-tier rate limit
    return reply


def test_hybrid_memory_recalls_early_detail_after_summarization():
    settings = load_settings()
    llm = build_llm(settings)
    memory = HybridMemoryStrategy(llm=llm, max_token_limit=300)  # low limit forces summarization
    session = ChatSession(llm=llm, memory=memory)

    _send(session, SECRET_DETAIL)

    for i in range(34):  # ~35 filler turns to cross the token limit
        _send(session, FILLER_TOPICS[i % len(FILLER_TOPICS)])

    stats = session.stats()
    assert stats["has_summary"], "expected summarization to have triggered by now"

    answer = _send(session, RECALL_QUESTION)
    assert "nimbus" in answer.lower(), f"expected recall of dog's name, got: {answer}"


def test_buffer_memory_does_not_crash_on_long_conversation():
    settings = load_settings()
    llm = build_llm(settings)
    memory = BufferMemoryStrategy()
    session = ChatSession(llm=llm, memory=memory)

    _send(session, SECRET_DETAIL)
    for i in range(34):
        _send(session, FILLER_TOPICS[i % len(FILLER_TOPICS)])

    answer = _send(session, RECALL_QUESTION)
    assert "nimbus" in answer.lower()

    stats = session.stats()
    assert stats["message_count"] == 72  # 36 turns, kept fully verbatim
