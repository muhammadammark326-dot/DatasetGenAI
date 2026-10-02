"""Schema for individual generated dataset examples."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from datasetgen.schemas.validation import ValidationResult


class GeneratedExample(BaseModel):
    """A single generated data item along with its provenance and validation history."""

    example_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any]
    topic: str
    difficulty: str
    is_valid: bool = False
    validation_history: List[ValidationResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_validation(self, result: ValidationResult) -> None:
        self.validation_history.append(result)
        self.is_valid = all(v.is_accepted for v in self.validation_history)
