import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: Optional[str]
    groq_model: str
    memory_max_token_limit: int


def load_settings() -> Settings:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to .env")

    return Settings(
        groq_api_key=groq_api_key,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        memory_max_token_limit=int(os.getenv("MEMORY_MAX_TOKEN_LIMIT", "1000")),
    )
