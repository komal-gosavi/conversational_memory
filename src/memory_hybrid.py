from typing import List
import tiktoken
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string

_encoding = None
_encoding_failed = False


def _count_tokens(messages: List[BaseMessage]) -> int:
    global _encoding, _encoding_failed
    text = get_buffer_string(messages)

    if not _encoding_failed and _encoding is None:
        try:
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding_failed = True

    if _encoding is not None:
        return len(_encoding.encode(text))

    return max(1, len(text) // 4)  # offline fallback


class _TiktokenSummaryBufferMemory(ConversationSummaryBufferMemory):
    def prune(self) -> None:
        buffer = self.chat_memory.messages
        curr_len = _count_tokens(buffer)
        if curr_len > self.max_token_limit:
            pruned = []
            while curr_len > self.max_token_limit:
                pruned.append(buffer.pop(0))
                curr_len = _count_tokens(buffer)
            self.moving_summary_buffer = self.predict_new_summary(pruned, self.moving_summary_buffer)


class HybridMemoryStrategy:
    name = "hybrid_summary_buffer"

    def __init__(self, llm: BaseLanguageModel, max_token_limit: int) -> None:
        self._memory = _TiktokenSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            return_messages=True,
        )

    def save_turn(self, human_input: str, ai_output: str) -> None:
        self._memory.save_context({"input": human_input}, {"output": ai_output})

    def get_context_messages(self) -> List[BaseMessage]:
        return self._memory.load_memory_variables({})["history"]

    def get_stats(self) -> dict:
        messages = self._memory.chat_memory.messages
        summary = self._memory.moving_summary_buffer
        return {
            "strategy": self.name,
            "message_count": len(messages),
            "has_summary": bool(summary),
            "summary_text": summary or None,
        }

    def clear(self) -> None:
        self._memory.clear()
