"""Registry for registering and instantiating validator pipelines."""

from __future__ import annotations

from typing import Dict, List, Type

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.validators.base import BaseValidator
from datasetgen.validators.classification import ClassificationValidator
from datasetgen.validators.code_sandbox import CodeSandboxValidator
from datasetgen.validators.constraints import ConstraintValidator
from datasetgen.validators.context_qa import ContextQAValidator
from datasetgen.validators.dedup import DedupValidator
from datasetgen.validators.json_extraction import JSONExtractionValidator
from datasetgen.validators.math_validator import MathValidator
from datasetgen.validators.physics import PhysicsNumericValidator
from datasetgen.validators.schema import SchemaValidator
from datasetgen.validators.semantic import SemanticJudgeValidator


class ValidatorRegistry:
    """Manages active validators and builds pipeline chains for blueprints."""

    _registry: Dict[str, Type[BaseValidator]] = {
        "schema": SchemaValidator,
        "dedup": DedupValidator,
        "constraints": ConstraintValidator,
        "physics_numeric": PhysicsNumericValidator,
        "math_symbolic": MathValidator,
        "classification": ClassificationValidator,
        "code_sandbox": CodeSandboxValidator,
        "context_qa": ContextQAValidator,
        "json_extraction": JSONExtractionValidator,
        "semantic_judge": SemanticJudgeValidator,
    }

    @classmethod
    def register(cls, name: str, validator_cls: Type[BaseValidator]) -> None:
        cls._registry[name] = validator_cls

    @classmethod
    def get_validator(cls, name: str) -> BaseValidator:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown validator '{name}'. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[name]()

    @classmethod
    def build_chain(cls, blueprint: Blueprint) -> List[BaseValidator]:
        """Instantiate all configured validators in prioritized order."""
        chain: List[BaseValidator] = []
        rules = list(blueprint.validation_rules)

        # 1. Enforce that schema check always runs first
        if "schema" in rules:
            chain.append(cls.get_validator("schema"))
            rules.remove("schema")

        # 2. Constraints second
        if "constraints" in rules:
            chain.append(cls.get_validator("constraints"))
            rules.remove("constraints")

        # 3. Dedup third
        if "dedup" in rules:
            chain.append(cls.get_validator("dedup"))
            rules.remove("dedup")

        # 4. Domain & Task-specific validators next
        for rule in rules:
            if rule in cls._registry:
                chain.append(cls.get_validator(rule))

        return chain
