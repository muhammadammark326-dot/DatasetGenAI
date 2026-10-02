"""Exact and near-duplicate detection validator."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set, Union

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


def _stem(word: str) -> str:
    """Lightweight suffix stripping for lexical matching."""
    return re.sub(r"(?:ing|es|ed|s)$", "", word)


def _tokenize(text: str) -> Set[str]:
    """Tokenize, normalize, and stem text into a token set, keeping numbers and content."""
    clean = re.sub(r"[^\w\s\.]", " ", text.lower())
    tokens = set()
    for w in clean.split():
        if len(w) > 0:
            if re.match(r"^\d+(?:\.\d+)?$", w):
                tokens.add(f"num_{w}")
            elif len(w) > 1:
                tokens.add(_stem(w))
    return tokens


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class DedupValidator(BaseValidator):
    """Enforces exact uniqueness and prevents semantic/syntactic near-duplicates."""

    name: str = "dedup"
    version: str = "v1"

    def __init__(self, near_dup_threshold: float = 0.85) -> None:
        self.near_dup_threshold = near_dup_threshold
        self.exact_hashes: Set[str] = set()
        self.seen_token_sets: List[Set[str]] = []

    def reset(self) -> None:
        """Clear cache for a fresh dataset run."""
        self.exact_hashes.clear()
        self.seen_token_sets.clear()

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)

        # Identify primary text content
        primary_text = ""
        for key in ("question", "prompt", "input", "text"):
            if key in data and isinstance(data[key], str):
                primary_text = data[key].strip()
                break

        if not primary_text:
            primary_text = " ".join(str(v) for v in data.values() if isinstance(v, str))

        # 1. Exact Duplicate Check (SHA256 of normalized text)
        normalized = re.sub(r"\s+", " ", primary_text.lower()).strip()
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if text_hash in self.exact_hashes:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Exact duplicate detected."],
                metadata={"duplicate_type": "exact", "hash": text_hash},
            )

        # 2. Near-Duplicate Check
        tokens = _tokenize(primary_text)
        for prev_tokens in self.seen_token_sets:
            sim = _jaccard_similarity(tokens, prev_tokens)
            if sim >= self.near_dup_threshold:
                return ValidationResult(
                    status="reject",
                    validator=self.identifier,
                    score=float(round(1.0 - sim, 4)),
                    errors=[f"Near-duplicate detected (similarity: {sim:.2f} >= {self.near_dup_threshold})"],
                    metadata={"duplicate_type": "near", "similarity": round(sim, 4)},
                )

        # Pass - record in seen set
        self.exact_hashes.add(text_hash)
        self.seen_token_sets.append(tokens)

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=1.0,
            errors=[],
        )
