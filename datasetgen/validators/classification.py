"""Classification validator ensuring label adherence and distribution consistency."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Union

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


class ClassificationValidator(BaseValidator):
    """Enforces that classification labels belong strictly to the allowed class set."""

    name: str = "classification"
    version: str = "v1"

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)

        # 1. Identify label field
        label_val = None
        label_key = None
        for key in ("label", "category", "class", "sentiment", "target"):
            if key in data:
                label_val = data[key]
                label_key = key
                break

        if label_val is None:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Classification example missing label field ('label', 'category', 'class')."],
            )

        # 2. Extract allowed classes from blueprint fields or topics
        allowed_classes: Set[str] = set()
        for f in blueprint.fields:
            if f.name == label_key and f.enum_values:
                allowed_classes = set(f.enum_values)
                break

        if not allowed_classes:
            # Fall back to blueprint topics or metadata
            allowed_classes = set(blueprint.topics.keys())

        # Normalize comparison
        str_val = str(label_val).strip()
        matched = False
        for allowed in allowed_classes:
            if str_val.lower() == str(allowed).lower():
                matched = True
                break

        if not matched:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=[
                    f"Label '{str_val}' is not in allowed classes: {sorted(list(allowed_classes))}"
                ],
                metadata={"allowed": list(allowed_classes), "received": str_val},
            )

        # 3. Check for input text presence
        input_text = ""
        for key in ("text", "input", "sentence", "premise", "content"):
            if key in data and isinstance(data[key], str):
                input_text = data[key].strip()
                break

        if not input_text:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Classification example missing non-empty input text."],
            )

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=1.0,
            errors=[],
            metadata={"label": str_val, "class_valid": True},
        )
