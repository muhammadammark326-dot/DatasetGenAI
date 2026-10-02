"""Blueprint builder orchestrating heuristic and LLM extraction."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from datasetgen.planner.extractors import (
    detect_domain_and_tasks,
    detect_missing_info,
    extract_count,
    extract_format,
    extract_language,
)
from datasetgen.providers.base import LLMProvider
from datasetgen.schemas.blueprint import Blueprint, BlueprintField


class BlueprintBuilder:
    """Creates a validated, immutable Blueprint from a natural-language request."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def build_blueprint(
        self,
        request: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Blueprint:
        """Analyze request and return an immutable Blueprint."""
        overrides = overrides or {}

        # 1. Deterministic extractions
        count = extract_count(request) or 100
        language = extract_language(request)
        out_format = extract_format(request)
        domain, task_type, subtopics = detect_domain_and_tasks(request)

        # 2. Build prompt for provider
        planner_prompt = f"""You are a master dataset architect. Convert this request into a structured Dataset Blueprint JSON:
Request: "{request}"

Requirements:
- Domain: {domain}
- Total count: {count}
- Language: {language}
- Output format: {out_format}
- Subtopics: {subtopics}

Return ONLY valid JSON matching this schema:
{{
  "dataset_name": "name_of_dataset",
  "dataset_version": "1.0.0",
  "blueprint_version": "1",
  "task_type": "{task_type}",
  "domain": "{domain}",
  "subdomains": {json.dumps(subtopics)},
  "education_level": "high_school",
  "language": "{language}",
  "number_of_examples": {count},
  "topics": {{"mechanics": 0.3, "thermodynamics": 0.2, "electricity": 0.2, "waves": 0.15, "optics": 0.15}},
  "difficulty_distribution": {{"beginner": 0.4, "intermediate": 0.4, "advanced": 0.2}},
  "fields": [
    {{"name": "question", "type": "string"}},
    {{"name": "answer", "type": "string"}},
    {{"name": "explanation", "type": "string"}},
    {{"name": "topic", "type": "string"}},
    {{"name": "difficulty", "type": "enum", "enum_values": ["beginner", "intermediate", "advanced"]}}
  ],
  "style_constraints": ["Include SI units", "Clear questions"],
  "safety_constraints": [],
  "output_format": "{out_format}",
  "validation_rules": ["schema", "dedup", "constraints", "physics_numeric"],
  "max_retries": 3,
  "random_seed": 42,
  "generator_model": "{self.provider.model_name}",
  "prompt_version": "v1"
}}"""

        response = self.provider.generate(planner_prompt, temperature=0.2)
        raw_text = response.text

        # Parse JSON from response
        blueprint_dict = self._parse_json(raw_text)

        # Merge deterministic values and overrides
        blueprint_dict["number_of_examples"] = overrides.get("number_of_examples", count)
        blueprint_dict["output_format"] = overrides.get("output_format", out_format)
        blueprint_dict["generator_model"] = self.provider.model_name

        if overrides:
            for k, v in overrides.items():
                blueprint_dict[k] = v

        # Normalize topic fractions to sum exactly to 1.0
        topics = blueprint_dict.get("topics", {})
        total_t = sum(topics.values())
        if total_t > 0:
            blueprint_dict["topics"] = {k: round(v / total_t, 4) for k, v in topics.items()}
            # Adjust remainder to last key to guarantee sum == 1.0
            last_key = list(blueprint_dict["topics"].keys())[-1]
            diff = 1.0 - sum(blueprint_dict["topics"].values())
            blueprint_dict["topics"][last_key] = round(blueprint_dict["topics"][last_key] + diff, 4)

        # Normalize difficulty distribution
        diffs = blueprint_dict.get("difficulty_distribution", {})
        total_d = sum(diffs.values())
        if total_d > 0:
            blueprint_dict["difficulty_distribution"] = {k: round(v / total_d, 4) for k, v in diffs.items()}
            last_k = list(blueprint_dict["difficulty_distribution"].keys())[-1]
            diff = 1.0 - sum(blueprint_dict["difficulty_distribution"].values())
            blueprint_dict["difficulty_distribution"][last_k] = round(
                blueprint_dict["difficulty_distribution"][last_k] + diff, 4
            )

        # Assign validation rules if domain is physics
        if domain == "physics" and "physics_numeric" not in blueprint_dict.get("validation_rules", []):
            rules = blueprint_dict.get("validation_rules", [])
            rules.append("physics_numeric")
            blueprint_dict["validation_rules"] = list(set(rules))

        return Blueprint.model_validate(blueprint_dict)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Safely extract JSON block from model response."""
        text = text.strip()
        # Look for ```json ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            m2 = re.search(r"(\{.*\})", text, re.DOTALL)
            if m2:
                text = m2.group(1)

        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"Failed to parse Blueprint JSON from model response: {e}\nRaw: {text[:200]}")
