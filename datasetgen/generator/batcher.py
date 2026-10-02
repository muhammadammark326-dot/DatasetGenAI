"""Batch generation coordinator with prompt versioning and caching."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from datasetgen.config.settings import get_settings
from datasetgen.providers.base import LLMProvider, LLMResponse
from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.storage.cache import GenerationCache


class BatchGenerator:
    """Coordinates generating structured batches of examples matching blueprint constraints."""

    def __init__(self, provider: LLMProvider, cache: Optional[GenerationCache] = None) -> None:
        self.provider = provider
        self.cache = cache or GenerationCache()

    def generate_batch(
        self,
        blueprint: Blueprint,
        topic: str,
        difficulty: str,
        count: int,
        batch_index: int,
    ) -> Tuple[List[GeneratedExample], LLMResponse]:
        """Generate a batch of examples targeted at a specific bucket."""
        # 1. Check cache
        cache_key = self.cache.compute_key(
            blueprint_dict=blueprint.model_dump(mode="json"),
            model=blueprint.generator_model,
            prompt_version=blueprint.prompt_version,
            batch_index=batch_index,
            seed=blueprint.random_seed + batch_index,
        )

        cached_data = self.cache.get(cache_key)
        if cached_data:
            examples = [
                GeneratedExample.model_validate(item) for item in cached_data["examples"]
            ]
            dummy_resp = LLMResponse(
                text="[cached]",
                model=blueprint.generator_model,
                metadata={"from_cache": True},
            )
            return examples, dummy_resp

        # 2. Build prompt
        prompt = self._build_prompt(blueprint, topic, difficulty, count)

        # 3. Call provider
        response = self.provider.generate(
            prompt,
            temperature=0.7,
            max_tokens=2048,
        )

        # 4. Parse examples
        parsed_items = self._parse_batch_json(response.text)

        examples: List[GeneratedExample] = []
        for item in parsed_items:
            # Ensure topic and difficulty are set
            item["topic"] = item.get("topic", topic)
            item["difficulty"] = item.get("difficulty", difficulty)
            ex = GeneratedExample(
                data=item,
                topic=item["topic"],
                difficulty=item["difficulty"],
                metadata={"batch_index": batch_index, "seed": blueprint.random_seed},
            )
            examples.append(ex)

        # 5. Store in cache
        cache_payload = {
            "examples": [e.model_dump(mode="json") for e in examples],
            "batch_index": batch_index,
        }
        self.cache.set(cache_key, cache_payload)

        return examples, response

    def _build_prompt(self, blueprint: Blueprint, topic: str, difficulty: str, count: int) -> str:
        field_descriptions = [
            f"- {f.name} ({f.type}): {f.description or 'field value'}"
            for f in blueprint.fields
        ]
        fields_str = "\n".join(field_descriptions)

        return f"""Generate exactly {count} examples for the following dataset:
Dataset: {blueprint.dataset_name}
Domain: {blueprint.domain}
Topic: {topic}
Difficulty: {difficulty}
Education Level: {blueprint.education_level or 'standard'}
Language: {blueprint.language}

Required JSON Fields for each example:
{fields_str}

Style & Constraints:
{chr(10).join(f"- {c}" for c in blueprint.style_constraints)}

Output strictly valid JSON with the format:
{{
  "examples": [
    {{ ... }},
    {{ ... }}
  ]
}}"""

    def _parse_batch_json(self, text: str) -> List[Dict[str, Any]]:
        """Extract array of examples from response."""
        text = text.strip()
        m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            m2 = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if m2:
                text = m2.group(1)

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "examples" in parsed:
                return parsed["examples"]
            elif isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            return []
        except Exception:
            return []
