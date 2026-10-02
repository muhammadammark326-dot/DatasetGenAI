"""Base validator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult


class BaseValidator(ABC):
    """Abstract base class for all deterministic and semantic validators."""

    name: str = "base"
    version: str = "v1"

    @property
    def identifier(self) -> str:
        return f"{self.name}_{self.version}"

    @abstractmethod
    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        """Validate an example against the blueprint.

        Returns ValidationResult with 'accept' or 'reject' and reasons.
        """
        pass

    @staticmethod
    def _extract_data(example: Union[Dict[str, Any], GeneratedExample]) -> Dict[str, Any]:
        """Normalize input to a plain dictionary of example fields."""
        if isinstance(example, GeneratedExample):
            return example.data
        return example
