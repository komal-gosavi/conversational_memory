from typing import List
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage


class BufferMemoryStrategy:
    name = "buffer"

    def __init__(self) -> None:
        self._memory = ConversationBufferMemory(return_messages=True)

    def save_turn(self, human_input: str, ai_output: str) -> None:
        self._memory.save_context({"input": human_input}, {"output": ai_output})

    def get_context_messages(self) -> List[BaseMessage]:
        return self._memory.load_memory_variables({})["history"]

    def get_stats(self) -> dict:
        messages = self._memory.chat_memory.messages
        return {
            "strategy": self.name,
            "message_count": len(messages),
            "has_summary": False,
            "summary_text": None,
        }

    def clear(self) -> None:
        self._memory.clear()
