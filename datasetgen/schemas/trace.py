"""Trace schema recording full generation lifecycle and decision telemetry."""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult


class Trace(BaseModel):
    """Full decision trace for dataset generation runs.

    This is the core training asset for downstream model fine-tuning.
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_request: str
    blueprint: Blueprint
    blueprint_version: str = "1"
    generator_model: str
    prompt_version: str = "v1"
    batch_id: Optional[str] = None
    examples: List[GeneratedExample] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    rejected_examples: List[GeneratedExample] = Field(default_factory=list)
    regeneration_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    human_corrections: List[Dict[str, Any]] = Field(default_factory=list)
    final_examples: List[GeneratedExample] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
