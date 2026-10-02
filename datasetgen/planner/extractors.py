"""Heuristic and regex extractors for prompt decomposition."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def extract_count(text: str) -> Optional[int]:
    """Extract requested number of examples (e.g. '100', '5k', '50,000')."""
    # 50k or 50K
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*[kK]\b", text)
    if k_match:
        return int(float(k_match.group(1)) * 1000)

    # 50,000 or 100
    num_match = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d+)\s*(?:examples|questions|items|samples|problems)?\b", text, re.I)
    if num_match:
        clean = num_match.group(1).replace(",", "")
        try:
            val = int(clean)
            if val > 0:
                return val
        except ValueError:
            pass
    return None


def extract_language(text: str) -> str:
    """Detect specified language in request."""
    langs = ["english", "urdu", "spanish", "french", "german", "chinese", "arabic", "hindi"]
    for lang in langs:
        if re.search(rf"\b{lang}\b", text, re.I):
            return lang.capitalize()
    return "English"


def extract_format(text: str) -> str:
    """Detect specified output format."""
    if "csv" in text.lower():
        return "csv"
    if "parquet" in text.lower():
        return "parquet"
    return "jsonl"


def detect_domain_and_tasks(text: str) -> Tuple[str, str, List[str]]:
    """Infer domain, task_type, and subtopics."""
    lower = text.lower()
    if "physic" in lower:
        return "physics", "question_answering", ["mechanics", "thermodynamics", "electricity", "waves", "optics"]
    elif "math" in lower or "algebra" in lower or "calculus" in lower:
        return "mathematics", "question_answering", ["algebra", "geometry", "calculus", "statistics"]
    elif "cod" in lower or "python" in lower or "programming" in lower:
        return "coding", "code_generation", ["functions", "algorithms", "data_structures"]
    elif "classif" in lower:
        return "classification", "classification", ["positive", "negative", "neutral"]
    return "general", "question_answering", ["general"]


def detect_missing_info(text: str) -> List[str]:
    """Check for missing or ambiguous requirements that would benefit from user clarification."""
    missing: List[str] = []
    if extract_count(text) is None:
        missing.append("Target example count was not specified (defaulting to 100).")
    if not any(k in text.lower() for k in ("beginner", "intermediate", "advanced", "grade", "level", "school")):
        missing.append("Target education/difficulty level was not explicitly stated.")
    return missing
