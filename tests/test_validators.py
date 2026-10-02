"""Unit tests for programmatic and domain validators."""

import pytest

from datasetgen.schemas.blueprint import Blueprint, BlueprintField
from datasetgen.validators.constraints import ConstraintValidator
from datasetgen.validators.dedup import DedupValidator
from datasetgen.validators.math_validator import MathValidator
from datasetgen.validators.physics import PhysicsNumericValidator
from datasetgen.validators.schema import SchemaValidator


@pytest.fixture
def sample_blueprint():
    return Blueprint(
        dataset_name="physics_test",
        number_of_examples=10,
        domain="physics",
        topics={"mechanics": 1.0},
        difficulty_distribution={"beginner": 1.0},
        fields=[
            BlueprintField(name="question", type="string"),
            BlueprintField(name="answer", type="string"),
            BlueprintField(name="difficulty", type="enum", enum_values=["beginner", "advanced"]),
        ],
    )


def test_schema_validator(sample_blueprint):
    val = SchemaValidator()

    # Valid example
    ok_example = {"question": "What is mass?", "answer": "Property of matter", "difficulty": "beginner"}
    res = val.validate(ok_example, sample_blueprint)
    assert res.is_accepted is True

    # Missing field
    missing_example = {"question": "What is mass?", "difficulty": "beginner"}
    res_bad = val.validate(missing_example, sample_blueprint)
    assert res_bad.is_accepted is False
    assert any("Missing required field: 'answer'" in err for err in res_bad.errors)

    # Invalid enum
    bad_enum = {"question": "What is mass?", "answer": "Stuff", "difficulty": "expert"}
    res_enum = val.validate(bad_enum, sample_blueprint)
    assert res_enum.is_accepted is False
    assert any("not in allowed enum" in err for err in res_enum.errors)


def test_dedup_validator(sample_blueprint):
    val = DedupValidator(near_dup_threshold=0.8)

    ex1 = {"question": "A 5 kg block accelerates at 2 m/s^2. What is the force?", "answer": "10 N"}
    res1 = val.validate(ex1, sample_blueprint)
    assert res1.is_accepted is True

    # Exact duplicate
    res_exact = val.validate(ex1, sample_blueprint)
    assert res_exact.is_accepted is False
    assert "Exact duplicate detected" in res_exact.errors[0]

    # Near duplicate
    ex_near = {"question": "A 5 kg block is accelerating at 2 m/s^2. What is the net force?", "answer": "10 N"}
    res_near = val.validate(ex_near, sample_blueprint)
    assert res_near.is_accepted is False
    assert "Near-duplicate detected" in res_near.errors[0]

    # Distinct problem
    ex_distinct = {"question": "Calculate the current across a 10 ohm resistor with 50 V DC.", "answer": "5 A"}
    res_distinct = val.validate(ex_distinct, sample_blueprint)
    assert res_distinct.is_accepted is True


def test_physics_numeric_validator(sample_blueprint):
    val = PhysicsNumericValidator()

    # 1. Newton's 2nd Law correct
    correct_mechanics = {
        "question": "A block with mass 5 kg accelerates across a frictionless surface at 10 m/s^2. What is the net force?",
        "answer": "50.0 N",
        "topic": "mechanics",
    }
    res_ok = val.validate(correct_mechanics, sample_blueprint)
    assert res_ok.is_accepted is True

    # 2. Newton's 2nd Law incorrect (calculated 50, answer 75)
    wrong_mechanics = {
        "question": "A block with mass 5 kg accelerates across a frictionless surface at 10 m/s^2. What is the net force?",
        "answer": "75.0 N",
        "topic": "mechanics",
    }
    res_bad = val.validate(wrong_mechanics, sample_blueprint)
    assert res_bad.is_accepted is False
    assert "does not match generated answer" in res_bad.errors[0]


def test_math_validator(sample_blueprint):
    val = MathValidator()

    # Correct solution 2x + 3 = 13 -> x = 5
    correct_math = {
        "question": "Solve the algebraic equation: 2*x + 3 = 13. Find the value of x.",
        "answer": "x = 5.0",
    }
    res_ok = val.validate(correct_math, sample_blueprint)
    assert res_ok.is_accepted is True

    # Wrong solution x = 7
    wrong_math = {
        "question": "Solve the algebraic equation: 2*x + 3 = 13. Find the value of x.",
        "answer": "x = 7.0",
    }
    res_bad = val.validate(wrong_math, sample_blueprint)
    assert res_bad.is_accepted is False
    assert "does not match SymPy solution" in res_bad.errors[0]


def test_constraint_validator(sample_blueprint):
    val = ConstraintValidator()

    # Placeholder phrasing
    evasive = {"question": "As an AI language model, here is physics.", "answer": "10"}
    res_evasive = val.validate(evasive, sample_blueprint)
    assert res_evasive.is_accepted is False

    # Negative mass
    bad_phys = {"question": "A ball with mass of -5 kg is dropped.", "answer": "10"}
    res_neg = val.validate(bad_phys, sample_blueprint)
    assert res_neg.is_accepted is False
