"""Registry for registering and instantiating validator pipelines."""

from __future__ import annotations

from typing import Dict, List, Type

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.validators.base import BaseValidator
from datasetgen.validators.constraints import ConstraintValidator
from datasetgen.validators.dedup import DedupValidator
from datasetgen.validators.math_validator import MathValidator
from datasetgen.validators.physics import PhysicsNumericValidator
from datasetgen.validators.schema import SchemaValidator


class ValidatorRegistry:
    """Manages active validators and builds pipeline chains for blueprints."""

    _registry: Dict[str, Type[BaseValidator]] = {
        "schema": SchemaValidator,
        "dedup": DedupValidator,
        "constraints": ConstraintValidator,
        "physics_numeric": PhysicsNumericValidator,
        "math_symbolic": MathValidator,
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

        # Enforce that schema check always runs first if present
        if "schema" in rules:
            chain.append(cls.get_validator("schema"))
            rules.remove("schema")

        # Constraints second
        if "constraints" in rules:
            chain.append(cls.get_validator("constraints"))
            rules.remove("constraints")

        # Dedup third
        if "dedup" in rules:
            chain.append(cls.get_validator("dedup"))
            rules.remove("dedup")

        # Domain validators next
        for rule in rules:
            chain.append(cls.get_validator(rule))

        return chain
