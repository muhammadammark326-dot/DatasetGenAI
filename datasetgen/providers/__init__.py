"""LLM Provider abstraction and implementations."""

from datasetgen.providers.base import LLMProvider, LLMResponse, LLMUsage
from datasetgen.providers.gemini import GeminiProvider
from datasetgen.providers.hf_local import HuggingFaceLocalProvider
from datasetgen.providers.mock import MockProvider
from datasetgen.providers.openai_compat import OpenAICompatProvider


def get_provider(name: str = "mock", **kwargs) -> LLMProvider:
    """Factory helper to instantiate providers by name."""
    clean_name = name.lower().strip()
    if clean_name == "mock":
        return MockProvider(**kwargs)
    elif clean_name in ("openai", "openai_compat"):
        return OpenAICompatProvider(**kwargs)
    elif clean_name == "gemini":
        return GeminiProvider(**kwargs)
    elif clean_name in ("hf_local", "local", "qlora", "huggingface"):
        return HuggingFaceLocalProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider '{name}'. Available: mock, openai, gemini, hf_local, qlora")


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMUsage",
    "MockProvider",
    "OpenAICompatProvider",
    "GeminiProvider",
    "HuggingFaceLocalProvider",
    "get_provider",
]

