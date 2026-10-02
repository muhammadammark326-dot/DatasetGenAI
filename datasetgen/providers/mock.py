"""Mock LLM Provider for offline, deterministic testing and development."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional

from datasetgen.providers.base import LLMProvider, LLMResponse, LLMUsage


class MockProvider(LLMProvider):
    """Deterministic mock provider that simulates LLM generation and planning.

    Supports configurable error rates to test schema, duplicate, domain, and constraint validators.
    """

    def __init__(
        self,
        model_name: str = "mock-generator-v1",
        seed: int = 42,
        error_rate: float = 0.15,
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

        # Check if the prompt is asking for a Blueprint / Plan
        if "blueprint" in prompt.lower() or "planner" in prompt.lower() or "extract" in prompt.lower():
            response_text = self._mock_blueprint_response(prompt)
        # Check if the prompt is asking for a batch of examples
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
        """Generate a mock valid Blueprint JSON."""
        count_match = re.search(r"Total count:\s*(\d+)", prompt)
        if not count_match:
            count_match = re.search(r"(\d+)", prompt)
        count = int(count_match.group(1)) if count_match else 100

        domain = "physics"
        if "math" in prompt.lower():
            domain = "mathematics"
        elif "code" in prompt.lower() or "coding" in prompt.lower():
            domain = "coding"
        elif "classif" in prompt.lower():
            domain = "classification"

        blueprint_data = {
            "dataset_name": f"{domain}_qa_dataset",
            "dataset_version": "1.0.0",
            "blueprint_version": "1",
            "task_type": "question_answering",
            "domain": domain,
            "subdomains": ["mechanics", "thermodynamics", "electricity", "waves", "optics"],
            "education_level": "high_school",
            "language": "English",
            "number_of_examples": count,
            "topics": {
                "mechanics": 0.3,
                "thermodynamics": 0.2,
                "electricity": 0.2,
                "waves": 0.15,
                "optics": 0.15,
            },
            "difficulty_distribution": {
                "beginner": 0.4,
                "intermediate": 0.4,
                "advanced": 0.2,
            },
            "fields": [
                {"name": "question", "type": "string", "description": "The physics problem statement"},
                {"name": "answer", "type": "string", "description": "The exact concise answer"},
                {"name": "explanation", "type": "string", "description": "Step by step calculation or derivation"},
                {"name": "topic", "type": "string", "description": "Physics subfield"},
                {"name": "difficulty", "type": "enum", "enum_values": ["beginner", "intermediate", "advanced"]},
            ],
            "style_constraints": ["Clear language", "Include SI units"],
            "safety_constraints": [],
            "output_format": "jsonl",
            "validation_rules": ["schema", "dedup", "constraints", "physics_numeric"],
            "max_retries": 3,
            "random_seed": self.seed,
            "generator_model": self.model_name,
            "prompt_version": "v1",
        }
        return json.dumps(blueprint_data, indent=2)

    def _mock_batch_response(self, prompt: str) -> str:
        """Generate a mock batch of examples in JSON format."""
        # Check prompt for requested count
        count_match = re.search(r"Generate exactly\s*(\d+)\s*examples", prompt, re.IGNORECASE)
        if not count_match:
            count_match = re.search(r"(\d+)\s+examples", prompt, re.IGNORECASE)
        n = int(count_match.group(1)) if count_match else 5

        topic_match = re.search(r"Topic:\s*([a-zA-Z_]+)", prompt, re.IGNORECASE)
        forced_topic = topic_match.group(1).lower() if topic_match else None

        diff_match = re.search(r"Difficulty:\s*([a-zA-Z_]+)", prompt, re.IGNORECASE)
        forced_diff = diff_match.group(1).lower() if diff_match else None

        examples = []
        topics = ["mechanics", "thermodynamics", "electricity", "waves", "optics"]
        diffs = ["beginner", "intermediate", "advanced"]

        for _ in range(n):
            topic = forced_topic if (forced_topic and forced_topic in topics) else self.rng.choice(topics)
            difficulty = forced_diff if (forced_diff and forced_diff in diffs) else self.rng.choice(diffs)
            item = self._generate_physics_item(topic, difficulty)

            # Injected errors for testing validators
            r = self.rng.random()
            if r < self.error_rate:
                error_type = self.rng.choice(["wrong_num", "duplicate", "bad_schema", "disallowed"])
                if error_type == "wrong_num":
                    item["answer"] = "9999.0 N"
                    item["explanation"] += " (Intentional domain failure for validator testing)"
                elif error_type == "duplicate":
                    item["question"] = "A standard 2 kg mass is accelerated at 3 m/s^2. What is the net force?"
                    item["answer"] = "6.0 N"
                elif error_type == "bad_schema":
                    item.pop("explanation", None)
                elif error_type == "disallowed":
                    item["question"] = "As an AI, a mass of 5 kg accelerates at 2 m/s^2."

            examples.append(item)

        return json.dumps({"examples": examples}, indent=2)

    def _generate_physics_item(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """Generate physically verifiable questions using diverse templates and objects."""
        objects = [
            "crate", "satellite", "locomotive", "cart", "vehicle", "payload", "sled",
            "drone", "rover", "capsule", "box", "glider", "elevator", "train car"
        ]
        obj = self.rng.choice(objects)

        if topic == "mechanics":
            m = self.rng.randint(2, 50)
            a = self.rng.randint(2, 15)
            f = m * a
            templates = [
                (
                    f"A {obj} with mass {m} kg accelerates across a frictionless surface at {a} m/s^2. What is the net force?",
                    f"{f:.1f} N",
                    f"Using Newton's second law F = m * a: F = {m} kg * {a} m/s^2 = {f} N."
                ),
                (
                    f"Determine the net force required to give a {m} kg {obj} an acceleration of {a} m/s^2.",
                    f"{f:.1f} N",
                    f"By F = m * a: F = {m} * {a} = {f} N."
                ),
                (
                    f"A propulsion engine exerts a constant force on a {m} kg {obj}, producing an acceleration of {a} m/s^2. Calculate the net force.",
                    f"{f:.1f} N",
                    f"F = m * a = {m} kg * {a} m/s^2 = {f} N."
                ),
                (
                    f"What net force causes a {m} kg {obj} to accelerate at {a} m/s^2 along a horizontal track?",
                    f"{f:.1f} N",
                    f"F = {m} * {a} = {f} N."
                ),
            ]
            question, answer, explanation = self.rng.choice(templates)

        elif topic == "electricity":
            v = self.rng.randint(4, 48) * 2
            r = self.rng.randint(2, 20)
            i_val = v / r
            circuits = ["heating coil", "incandescent lamp", "ceramic resistor", "potentiometer", "component"]
            c_name = self.rng.choice(circuits)
            templates = [
                (
                    f"A circuit contains a {c_name} of {r} ohms connected to a {v} V DC power supply. Calculate the current.",
                    f"{i_val:.1f} A",
                    f"Using Ohm's law I = V / R: I = {v} V / {r} ohms = {i_val:.1f} A."
                ),
                (
                    f"What electric current flows through a {r} ohms {c_name} subjected to a potential difference of {v} V?",
                    f"{i_val:.1f} A",
                    f"I = V / R = {v} / {r} = {i_val:.1f} A."
                ),
                (
                    f"Determine the current in amperes drawn by a {r} ohms {c_name} when wired across a {v} V battery.",
                    f"{i_val:.1f} A",
                    f"By Ohm's law I = {v} V / {r} ohms = {i_val:.1f} A."
                ),
            ]
            question, answer, explanation = self.rng.choice(templates)

        elif topic == "thermodynamics":
            m = self.rng.randint(1, 10)
            dt = self.rng.randint(5, 60)
            c = 4184
            q = m * c * dt
            templates = [
                (
                    f"How much heat energy is required to raise the temperature of {m} kg of water by {dt} K?",
                    f"{q:.1f} J",
                    f"Using Q = m * c * DeltaT: Q = {m} * 4184 * {dt} = {q:.1f} J."
                ),
                (
                    f"Calculate the thermal energy in Joules absorbed by {m} kg of water warmed by {dt} K.",
                    f"{q:.1f} J",
                    f"Q = m * c * DeltaT = {m} * 4184 * {dt} = {q:.1f} J."
                ),
                (
                    f"A laboratory calorimeter transfers heat to {m} kg of water, increasing its temperature by {dt} K. Find the energy.",
                    f"{q:.1f} J",
                    f"Q = {m} * 4184 * {dt} = {q:.1f} J."
                ),
            ]
            question, answer, explanation = self.rng.choice(templates)

        elif topic == "waves":
            freq = self.rng.randint(50, 1500)
            wavelength = self.rng.randint(1, 8)
            speed = freq * wavelength
            media = ["sound wave", "seismic wave", "acoustic pulse", "surface wave"]
            w_type = self.rng.choice(media)
            templates = [
                (
                    f"A {w_type} has a frequency of {freq} Hz and a wavelength of {wavelength} m. What is its speed?",
                    f"{speed:.1f} m/s",
                    f"Using v = f * lambda: v = {freq} Hz * {wavelength} m = {speed:.1f} m/s."
                ),
                (
                    f"Determine the speed of a {w_type} with frequency {freq} Hz and spatial wavelength of {wavelength} m.",
                    f"{speed:.1f} m/s",
                    f"v = f * lambda = {freq} * {wavelength} = {speed:.1f} m/s."
                ),
                (
                    f"Calculate the propagation velocity of a {w_type} oscillating at {freq} Hz with wavelength {wavelength} m.",
                    f"{speed:.1f} m/s",
                    f"v = {freq} * {wavelength} = {speed:.1f} m/s."
                ),
            ]
            question, answer, explanation = self.rng.choice(templates)

        else: # optics
            focal = self.rng.randint(10, 30)
            do = self.rng.randint(40, 90)
            di = (focal * do) / (do - focal)
            templates = [
                (
                    f"An object is placed {do} cm in front of a convex lens with focal length {focal} cm. Where is the image formed?",
                    f"{di:.2f} cm",
                    f"Using 1/f = 1/d_o + 1/d_i: d_i = ({focal} * {do}) / ({do} - {focal}) = {di:.2f} cm."
                ),
                (
                    f"Find the image distance when an illuminated slide is placed {do} cm before a thin convex lens of focal length {focal} cm.",
                    f"{di:.2f} cm",
                    f"1/di = 1/f - 1/do: di = ({focal} * {do}) / ({do - focal}) = {di:.2f} cm."
                ),
            ]
            question, answer, explanation = self.rng.choice(templates)

        return {
            "question": question,
            "answer": answer,
            "explanation": explanation,
            "topic": topic,
            "difficulty": difficulty,
        }
