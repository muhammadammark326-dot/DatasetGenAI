"""Unit tests for Pydantic v2 schemas."""

import pytest
from pydantic import ValidationError

from datasetgen.schemas.blueprint import Blueprint, BlueprintField
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.trace import Trace
from datasetgen.schemas.validation import ValidationResult


def test_blueprint_valid():
    bp = Blueprint(
        dataset_name="test_physics",
        number_of_examples=50,
        domain="physics",
        topics={"mechanics": 0.5, "optics": 0.5},
        difficulty_distribution={"beginner": 0.6, "advanced": 0.4},
        fields=[
            BlueprintField(name="question", type="string"),
            BlueprintField(name="answer", type="string"),
        ],
    )
    assert bp.dataset_name == "test_physics"
    assert bp.number_of_examples == 50


def test_blueprint_immutable():
    bp = Blueprint(
        dataset_name="test_immutable",
        number_of_examples=10,
        domain="physics",
        topics={"mechanics": 1.0},
        difficulty_distribution={"beginner": 1.0},
        fields=[BlueprintField(name="q", type="string")],
    )
    with pytest.raises(ValidationError):
        bp.number_of_examples = 20


def test_blueprint_invalid_distribution_sum():
    with pytest.raises(ValidationError):
        Blueprint(
            dataset_name="test_bad_sum",
            number_of_examples=10,
            domain="physics",
            topics={"mechanics": 0.3, "optics": 0.3}, # sums to 0.6 != 1.0
            difficulty_distribution={"beginner": 1.0},
            fields=[BlueprintField(name="q", type="string")],
        )


def test_validation_result():
    res_pass = ValidationResult(status="accept", validator="test_v1")
    assert res_pass.is_accepted is True

    res_fail = ValidationResult(status="reject", validator="test_v1", errors=["Invalid answer"])
    assert res_fail.is_accepted is False
    assert len(res_fail.errors) == 1
