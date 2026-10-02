"""Pydantic v2 Blueprint schema. Immutable once created."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BlueprintField(BaseModel):
    """Specification for a single field in the generated examples."""

    name: str
    type: Literal["string", "number", "integer", "boolean", "enum", "list", "dict"]
    enum_values: Optional[List[str]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    description: Optional[str] = None


class Blueprint(BaseModel):
    """Immutable contract defining all parameters of dataset generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_name: str
    dataset_version: str = "1.0.0"
    blueprint_version: str = "1"
    task_type: str = "question_answering"
    domain: str
    subdomains: List[str] = Field(default_factory=list)
    education_level: Optional[str] = None
    language: str = "English"
    number_of_examples: int = Field(gt=0)
    topics: Dict[str, float] = Field(
        description="Dictionary mapping topic name to fraction (sums to 1.0)"
    )
    difficulty_distribution: Dict[str, float] = Field(
        default_factory=lambda: {"beginner": 0.4, "intermediate": 0.4, "advanced": 0.2},
        description="Difficulty distribution fractions (sums to 1.0)",
    )
    fields: List[BlueprintField]
    style_constraints: List[str] = Field(default_factory=list)
    safety_constraints: List[str] = Field(default_factory=list)
    output_format: Literal["jsonl", "csv", "parquet"] = "jsonl"
    validation_rules: List[str] = Field(
        default_factory=lambda: ["schema", "dedup"]
    )
    max_retries: int = Field(default=3, ge=0)
    random_seed: int = Field(default=42)
    generator_model: str = "mock"
    prompt_version: str = "v1"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("topics", "difficulty_distribution")
    @classmethod
    def validate_distribution_sums(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Distribution dictionary cannot be empty.")
        total = sum(v.values())
        if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"Fractions must sum to 1.0 (got {total:.4f}). Distribution: {v}")
        return v
