"""Deterministic constraint validator."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


class ConstraintValidator(BaseValidator):
    """Enforces deterministic constraints such as non-negative physical values and style rules."""

    name: str = "constraints"
    version: str = "v1"

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)
        errors: List[str] = []

        # Check for placeholder / evasive text
        evasive_patterns = [
            r"as an ai",
            r"i cannot",
            r"as requested",
            r"lorem ipsum",
            r"todo",
            r"tbd",
        ]
        text_corpus = " ".join(str(v) for v in data.values()).lower()
        for pattern in evasive_patterns:
            if re.search(r"\b" + pattern + r"\b", text_corpus):
                errors.append(f"Contains disallowed placeholder / evasive phrasing: '{pattern}'")

        # Physical sanity: mass, frequency, resistance must not be negative
        neg_physical_pattern = r"(?:mass|resistance|frequency|wavelength|speed)\s*(?:of|=|is)\s*-\d+"
        if re.search(neg_physical_pattern, text_corpus):
            errors.append("Physical quantity has an impossible negative value.")

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
