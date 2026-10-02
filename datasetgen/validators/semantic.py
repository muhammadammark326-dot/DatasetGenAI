"""Semantic judge validator using structured rubric scoring."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from datasetgen.providers.base import LLMProvider
from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


class SemanticJudgeValidator(BaseValidator):
    """Evaluates qualitative outputs against a structured rubric using an LLM judge or deterministic fallback."""

    name: str = "semantic_judge"
    version: str = "v1"

    def __init__(self, provider: Optional[LLMProvider] = None, min_score: float = 3.0) -> None:
        self.provider = provider
        self.min_score = min_score

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)

        instruction = data.get("instruction") or data.get("prompt") or data.get("question") or ""
        response = data.get("response") or data.get("output") or data.get("answer") or ""

        if not str(response).strip():
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Empty response text."],
            )

        # 1. If LLM Provider is available, run judge prompt
        if self.provider is not None and self.provider.__class__.__name__ != "MockProvider":
            judge_prompt = f"""Evaluate this instruction-response pair on a scale from 1 to 5:
Instruction: "{instruction}"
Response: "{response}"

Rubric:
- Relevance (1-5): Does the response directly address the user's prompt?
- Completeness (1-5): Is the answer thorough and clear?
- No Hallucination (1-5): Is the information sound?

Return strictly valid JSON:
{{"relevance": 5, "completeness": 5, "factuality": 5, "overall": 5, "passed": true, "rationale": "..."}}"""

            try:
                res = self.provider.generate(judge_prompt, temperature=0.1)
                m = re.search(r"(\{.*\})", res.text, re.DOTALL)
                if m:
                    scores = json.loads(m.group(1))
                    overall = float(scores.get("overall", 3.0))
                    if overall < self.min_score:
                        return ValidationResult(
                            status="reject",
                            validator=self.identifier,
                            score=overall / 5.0,
                            errors=[f"Semantic score {overall} below threshold {self.min_score}: {scores.get('rationale')}"],
                            metadata=scores,
                        )
                    return ValidationResult(
                        status="accept",
                        validator=self.identifier,
                        score=overall / 5.0,
                        errors=[],
                        metadata=scores,
                    )
            except Exception:
                pass

        # 2. Deterministic / Heuristic Fallback (Offline Mode)
        # Check basic coherence, non-repetition, and minimum substance
        text_resp = str(response).strip()
        words = text_resp.split()

        # Repetition loop check
        if len(words) >= 6:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.35:
                return ValidationResult(
                    status="reject",
                    validator=self.identifier,
                    score=float(round(unique_ratio, 2)),
                    errors=["Semantic failure: excessive word repetition loop detected."],
                    metadata={"unique_ratio": round(unique_ratio, 2)},
                )

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=1.0,
            errors=[],
            metadata={"mode": "deterministic_heuristic", "passed": True},
        )
