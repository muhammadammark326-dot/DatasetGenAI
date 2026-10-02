"""Base abstraction for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """Token usage and cost statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""

    text: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, model_name: str = "default", **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text completion from a prompt."""
        pass
