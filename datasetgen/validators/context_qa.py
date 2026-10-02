"""Context-grounded QA validator."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Union

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


def _extract_content_tokens(text: str) -> Set[str]:
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    stop_words = {
        "a", "an", "the", "is", "was", "are", "were", "and", "or", "in", "on", "at",
        "to", "for", "of", "with", "by", "that", "this", "it", "from", "as"
    }
    return {w for w in clean.split() if len(w) > 2 and w not in stop_words}


class ContextQAValidator(BaseValidator):
    """Verifies that an answer in a reading-comprehension example is directly supported by its context."""

    name: str = "context_qa"
    version: str = "v1"

    def __init__(self, min_support_ratio: float = 0.50) -> None:
        self.min_support_ratio = min_support_ratio

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)

        context = data.get("context") or data.get("passage") or data.get("document") or ""
        question = data.get("question") or data.get("query") or ""
        answer = str(data.get("answer", "")).strip()

        if not context or not context.strip():
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Context-QA example missing 'context' passage."],
            )

        if not answer:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Context-QA example missing 'answer'."],
            )

        # 1. Exact Substring Containment Check
        clean_context = context.lower()
        clean_answer = answer.lower()
        if clean_answer in clean_context:
            return ValidationResult(
                status="accept",
                validator=self.identifier,
                score=1.0,
                errors=[],
                metadata={"grounding_type": "exact_span"},
            )

        # 2. Token Grounding Overlap Check
        answer_tokens = _extract_content_tokens(answer)
        if not answer_tokens:
            return ValidationResult(
                status="accept",
                validator=self.identifier,
                score=1.0,
                errors=[],
                metadata={"grounding_type": "short_answer"},
            )

        context_tokens = _extract_content_tokens(context)
        grounded_tokens = answer_tokens.intersection(context_tokens)
        support_ratio = len(grounded_tokens) / len(answer_tokens)

        if support_ratio < self.min_support_ratio:
            unsupported = list(answer_tokens - context_tokens)[:5]
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=float(round(support_ratio, 3)),
                errors=[
                    f"Answer not supported by context. Support ratio: {support_ratio:.2f} < {self.min_support_ratio} (unsupported tokens: {unsupported})"
                ],
                metadata={"support_ratio": round(support_ratio, 3), "unsupported": unsupported},
            )

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=float(round(support_ratio, 3)),
            errors=[],
            metadata={"support_ratio": round(support_ratio, 3), "grounding_type": "lexical_support"},
        )
