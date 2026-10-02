"""Schema validator for strict structural adherence."""

from __future__ import annotations

from typing import Any, Dict, List, Union

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


class SchemaValidator(BaseValidator):
    """Enforces presence, data types, enum boundaries, and length limits for blueprint fields."""

    name: str = "schema"
    version: str = "v1"

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)
        errors: List[str] = []

        if not isinstance(data, dict):
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Example is not a valid JSON dictionary."],
            )

        for field_spec in blueprint.fields:
            fname = field_spec.name
            if fname not in data or data[fname] is None:
                errors.append(f"Missing required field: '{fname}'")
                continue

            val = data[fname]

            # Type checking
            if field_spec.type == "string":
                if not isinstance(val, str):
                    errors.append(f"Field '{fname}' must be string, got {type(val).__name__}")
                else:
                    if not val.strip():
                        errors.append(f"Field '{fname}' cannot be blank")
                    if field_spec.min_length and len(val) < field_spec.min_length:
                        errors.append(
                            f"Field '{fname}' length {len(val)} < min_length {field_spec.min_length}"
                        )
                    if field_spec.max_length and len(val) > field_spec.max_length:
                        errors.append(
                            f"Field '{fname}' length {len(val)} > max_length {field_spec.max_length}"
                        )
            elif field_spec.type in ("number", "integer"):
                if field_spec.type == "integer" and not isinstance(val, int):
                    errors.append(f"Field '{fname}' must be integer, got {type(val).__name__}")
                elif not isinstance(val, (int, float)):
                    errors.append(f"Field '{fname}' must be number, got {type(val).__name__}")
            elif field_spec.type == "boolean":
                if not isinstance(val, bool):
                    errors.append(f"Field '{fname}' must be boolean, got {type(val).__name__}")
            elif field_spec.type == "enum":
                if field_spec.enum_values and str(val) not in field_spec.enum_values:
                    errors.append(
                        f"Field '{fname}' value '{val}' not in allowed enum: {field_spec.enum_values}"
                    )
            elif field_spec.type == "list":
                if not isinstance(val, list):
                    errors.append(f"Field '{fname}' must be a list, got {type(val).__name__}")

        if errors:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=errors,
            )

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=1.0,
            errors=[],
        )
