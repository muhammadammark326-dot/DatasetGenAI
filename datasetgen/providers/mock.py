"""Mock LLM Provider for offline, deterministic testing and multi-domain benchmarking."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional

from datasetgen.providers.base import LLMProvider, LLMResponse, LLMUsage


class MockProvider(LLMProvider):
    """Deterministic mock provider that simulates LLM generation across all dataset domains."""

    def __init__(
        self,
        model_name: str = "mock-generator-v1",
        seed: int = 42,
        error_rate: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.seed = seed
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self._counter = 0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._counter += 1

        if "blueprint" in prompt.lower() or "planner" in prompt.lower() or "extract" in prompt.lower():
            response_text = self._mock_blueprint_response(prompt)
        else:
            response_text = self._mock_batch_response(prompt)

        tokens = len(prompt.split()) + len(response_text.split())
        usage = LLMUsage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=tokens,
            estimated_cost_usd=0.0,
        )

        return LLMResponse(
            text=response_text,
            usage=usage,
            model=self.model_name,
            metadata={"seed": self.seed, "counter": self._counter},
        )

    def _mock_blueprint_response(self, prompt: str) -> str:
        """Generate a mock valid Blueprint JSON customized for the requested domain."""
        count_match = re.search(r"Total count:\s*(\d+)", prompt)
        if not count_match:
            count_match = re.search(r"(\d+)", prompt)
        count = int(count_match.group(1)) if count_match else 10

        p_low = prompt.lower()
        if "cod" in p_low:
            domain = "coding"
            task_type = "code_generation"
            subtopics = ["algorithms", "functions", "strings"]
            fields = [
                {"name": "code", "type": "string", "description": "Python function implementation"},
                {"name": "test_cases", "type": "string", "description": "Unit test assertions"},
                {"name": "topic", "type": "string"},
                {"name": "difficulty", "type": "enum", "enum_values": ["beginner", "intermediate", "advanced"]},
            ]
            rules = ["schema", "dedup", "code_sandbox"]
        elif "classif" in p_low or "sentiment" in p_low:
            domain = "classification"
            task_type = "classification"
            subtopics = ["positive", "negative"]
            fields = [
                {"name": "text", "type": "string", "description": "Text to classify"},
                {"name": "label", "type": "enum", "enum_values": ["positive", "negative"]},
                {"name": "topic", "type": "string"},
                {"name": "difficulty", "type": "enum", "enum_values": ["beginner", "intermediate", "advanced"]},
            ]
            rules = ["schema", "dedup", "classification"]
        elif "math" in p_low or "algebra" in p_low:
            domain = "mathematics"
            task_type = "question_answering"
            subtopics = ["algebra", "equations"]
            fields = [
                {"name": "question", "type": "string"},
                {"name": "answer", "type": "string"},
                {"name": "explanation", "type": "string"},
                {"name": "topic", "type": "string"},
                {"name": "difficulty", "type": "enum", "enum_values": ["beginner", "intermediate", "advanced"]},
            ]
            rules = ["schema", "dedup", "math_symbolic"]
        elif "comprehension" in p_low or "context" in p_low or "passage" in p_low:
            domain = "context_qa"
            task_type = "reading_comprehension"
            subtopics = ["science", "history", "geography"]
            fields = [
                {"name": "context", "type": "string"},
                {"name": "question", "type": "string"},
                {"name": "answer", "type": "string"},
                {"name": "topic", "type": "string"},
                {"name": "difficulty", "type": "enum", "enum_values": ["beginner", "intermediate", "advanced"]},
            ]
            rules = ["schema", "dedup", "context_qa"]
        else:
            domain = "physics"
            task_type = "question_answering"
            subtopics = ["mechanics", "electricity", "thermodynamics"]
            fields = [
                {"name": "question", "type": "string"},
                {"name": "answer", "type": "string"},
                {"name": "explanation", "type": "string"},
                {"name": "topic", "type": "string"},
                {"name": "difficulty", "type": "enum", "enum_values": ["beginner", "intermediate", "advanced"]},
            ]
            rules = ["schema", "dedup", "physics_numeric"]

        topics_dist = {s: round(1.0 / len(subtopics), 4) for s in subtopics}
        last_s = list(topics_dist.keys())[-1]
        topics_dist[last_s] = round(1.0 - sum(list(topics_dist.values())[:-1]), 4)

        blueprint_data = {
            "dataset_name": f"{domain}_dataset",
            "dataset_version": "1.0.0",
            "blueprint_version": "1",
            "task_type": task_type,
            "domain": domain,
            "subdomains": subtopics,
            "education_level": "standard",
            "language": "English",
            "number_of_examples": count,
            "topics": topics_dist,
            "difficulty_distribution": {
                "beginner": 0.4,
                "intermediate": 0.4,
                "advanced": 0.2,
            },
            "fields": fields,
            "style_constraints": ["Clear format"],
            "safety_constraints": [],
            "output_format": "jsonl",
            "validation_rules": rules,
            "max_retries": 3,
            "random_seed": self.seed,
            "generator_model": self.model_name,
            "prompt_version": "v1",
        }
        return json.dumps(blueprint_data, indent=2)

    def _mock_batch_response(self, prompt: str) -> str:
        """Generate a mock batch of examples corresponding to the prompt's domain."""
        count_match = re.search(r"Generate exactly\s*(\d+)\s*examples", prompt, re.IGNORECASE)
        if not count_match:
            count_match = re.search(r"(\d+)\s+examples", prompt, re.IGNORECASE)
        n = int(count_match.group(1)) if count_match else 5

        topic_match = re.search(r"Topic:\s*([a-zA-Z_]+)", prompt, re.IGNORECASE)
        forced_topic = topic_match.group(1).lower() if topic_match else "general"

        diff_match = re.search(r"Difficulty:\s*([a-zA-Z_]+)", prompt, re.IGNORECASE)
        forced_diff = diff_match.group(1).lower() if diff_match else "beginner"

        p_low = prompt.lower()
        examples = []

        for _ in range(n):
            if "domain: coding" in p_low or "code" in p_low:
                item = self._generate_code_item(forced_topic, forced_diff)
            elif "domain: classification" in p_low or "sentiment" in p_low:
                item = self._generate_classification_item(forced_topic, forced_diff)
            elif "domain: mathematics" in p_low or "algebra" in p_low:
                item = self._generate_math_item(forced_topic, forced_diff)
            elif "domain: context_qa" in p_low or "passage" in p_low:
                item = self._generate_context_qa_item(forced_topic, forced_diff)
            else:
                item = self._generate_physics_item(forced_topic, forced_diff)

            # Error injection
            if self.rng.random() < self.error_rate:
                if "code" in item:
                    item["code"] = "def broken(x):\n    return x + 999" # Assertion failure
                elif "label" in item:
                    item["label"] = "invalid_class_99" # Bad class
                elif "answer" in item:
                    item["answer"] = "9999.0"

            examples.append(item)

        return json.dumps({"examples": examples}, indent=2)

    def _generate_code_item(self, topic: str, difficulty: str) -> Dict[str, Any]:
        idx = self.rng.randint(1, 9999)
        funcs = [
            (f"def double_val_{idx}(x: int) -> int:\n    return x * 2", f"assert double_val_{idx}(4) == 8\nassert double_val_{idx}(0) == 0"),
            (f"def is_even_{idx}(n: int) -> bool:\n    return n % 2 == 0", f"assert is_even_{idx}(4) is True\nassert is_even_{idx}(7) is False"),
            (f"def add_lists_{idx}(a: list, b: list) -> list:\n    return a + b", f"assert add_lists_{idx}([1], [2]) == [1, 2]"),
            (f"def square_num_{idx}(n: int) -> int:\n    return n ** 2", f"assert square_num_{idx}(5) == 25\nassert square_num_{idx}(3) == 9"),
            (f"def greet_user_{idx}(name: str) -> str:\n    return f'Hello, {{name}}'", f"assert greet_user_{idx}('World') == 'Hello, World'"),
            (f"def count_chars_{idx}(s: str) -> int:\n    return len(s)", f"assert count_chars_{idx}('sample') == 6"),
            (f"def max_pair_{idx}(a: int, b: int) -> int:\n    return max(a, b)", f"assert max_pair_{idx}(10, 20) == 20"),
        ]
        code, test = self.rng.choice(funcs)
        return {
            "code": code,
            "test_cases": test,
            "topic": topic,
            "difficulty": difficulty,
        }

    def _generate_classification_item(self, topic: str, difficulty: str) -> Dict[str, Any]:
        idx = self.rng.randint(1, 9999)
        positives = [
            f"This product #{idx} exceeded all my expectations, truly wonderful.",
            f"The customer service for order #{idx} was fast, helpful, and very friendly.",
            f"An absolutely fantastic experience (#{idx}), five stars all the way.",
            f"Great quality build on item #{idx} and arrived earlier than anticipated.",
        ]
        negatives = [
            f"Terrible experience with unit #{idx}, broke after only two days of light use.",
            f"The delivery #{idx} was delayed by three weeks and customer support was unresponsive.",
            f"Very poor build quality on model #{idx} and does not match description.",
            f"Completely disappointed with #{idx}, will be returning and demanding a refund.",
        ]
        label = "positive" if (topic == "positive" or self.rng.random() > 0.5) else "negative"
        text = self.rng.choice(positives) if label == "positive" else self.rng.choice(negatives)
        return {
            "text": text,
            "label": label,
            "topic": topic,
            "difficulty": difficulty,
        }

    def _generate_context_qa_item(self, topic: str, difficulty: str) -> Dict[str, Any]:
        idx = self.rng.randint(1, 9999)
        passages = [
            (
                f"Photosynthesis in specimen #{idx} is the biochemical process by which plants use sunlight, water, and carbon dioxide to create oxygen and glucose.",
                f"What gas do plants produce as a byproduct in specimen #{idx}?",
                "Oxygen is produced as a byproduct of photosynthesis."
            ),
            (
                f"The Hubble Space Telescope mission #{idx} was launched into low Earth orbit in 1990 and remains in operation, capturing deep-space observations.",
                f"In what year was the Hubble Space Telescope mission #{idx} launched?",
                "The Hubble Space Telescope was launched in 1990."
            ),
            (
                f"The Pacific Ocean sector #{idx} is the largest and deepest of Earth's five oceanic divisions, extending from the Arctic to the Southern Ocean.",
                f"Which ocean is the largest and deepest according to survey #{idx}?",
                "The Pacific Ocean is the largest and deepest ocean on Earth."
            ),
        ]
        ctx, q, a = self.rng.choice(passages)
        return {
            "context": ctx,
            "question": q,
            "answer": a,
            "topic": topic,
            "difficulty": difficulty,
        }

    def _generate_math_item(self, topic: str, difficulty: str) -> Dict[str, Any]:
        a = self.rng.randint(2, 6)
        x = self.rng.randint(2, 9)
        b = self.rng.randint(1, 10)
        c = a * x + b
        return {
            "question": f"Solve the algebraic equation: {a}*x + {b} = {c}. Find the value of x.",
            "answer": f"x = {float(x)}",
            "explanation": f"Subtract {b} from {c} to get {c - b}, then divide by {a} to get x = {x}.",
            "topic": topic,
            "difficulty": difficulty,
        }

    def _generate_physics_item(self, topic: str, difficulty: str) -> Dict[str, Any]:
        objects = ["crate", "satellite", "locomotive", "cart", "vehicle", "payload", "sled", "drone", "rover", "capsule"]
        obj = self.rng.choice(objects)

        if "elec" in topic:
            v = self.rng.randint(4, 24) * 2
            r = self.rng.randint(2, 12)
            i_val = v / r
            question = f"A circuit contains a resistor of {r} ohms connected to a {v} V DC source. Calculate the current."
            answer = f"{i_val:.1f} A"
            explanation = f"Using Ohm's law I = V / R: I = {v} V / {r} ohms = {i_val:.1f} A."
        else:
            m = self.rng.randint(2, 30)
            a = self.rng.randint(2, 10)
            f = m * a
            question = f"A {obj} with mass {m} kg accelerates across a frictionless surface at {a} m/s^2. What is the net force?"
            answer = f"{f:.1f} N"
            explanation = f"Using Newton's second law F = m * a: F = {m} kg * {a} m/s^2 = {f} N."

        return {
            "question": question,
            "answer": answer,
            "explanation": explanation,
            "topic": topic,
            "difficulty": difficulty,
        }
