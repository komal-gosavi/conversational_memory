from langchain_core.language_models import BaseLanguageModel
from .config import Settings


def build_llm(settings: Settings) -> BaseLanguageModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
    )
