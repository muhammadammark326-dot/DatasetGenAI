"""JSON extraction and schema adherence validator."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


class JSONExtractionValidator(BaseValidator):
    """Validates that extracted information conforms to valid JSON structure and non-empty values."""

    name: str = "json_extraction"
    version: str = "v1"

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)

        extracted = data.get("extracted_json") or data.get("output") or data.get("json")

        if extracted is None:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Missing 'extracted_json' or 'output' field."],
            )

        # Parse string if not already dict
        parsed_dict = None
        if isinstance(extracted, str):
            try:
                parsed_dict = json.loads(extracted.strip())
            except Exception as e:
                return ValidationResult(
                    status="reject",
                    validator=self.identifier,
                    score=0.0,
                    errors=[f"Failed to parse extracted JSON string: {e}"],
                )
        elif isinstance(extracted, dict):
            parsed_dict = extracted
        else:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=[f"Expected JSON object or dict, got {type(extracted).__name__}"],
            )

        if not parsed_dict:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Extracted JSON object is empty."],
            )

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=1.0,
            errors=[],
            metadata={"parsed_keys": list(parsed_dict.keys())},
        )
