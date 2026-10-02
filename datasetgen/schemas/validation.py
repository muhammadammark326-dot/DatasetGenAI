"""Validation result schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of a single validator pass or composite validation."""

    status: Literal["accept", "reject"]
    validator: str = Field(description="Name and version of the validator, e.g. physics_numeric_v1")
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_accepted(self) -> bool:
        return self.status == "accept"
