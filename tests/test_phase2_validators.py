"""Unit tests for Phase 2 validators: Classification, Code Sandbox, Context-QA, JSON Extraction, and Semantic Judge."""

import pytest

from datasetgen.schemas.blueprint import Blueprint, BlueprintField
from datasetgen.validators.classification import ClassificationValidator
from datasetgen.validators.code_sandbox import CodeSandboxValidator
from datasetgen.validators.context_qa import ContextQAValidator
from datasetgen.validators.json_extraction import JSONExtractionValidator
from datasetgen.validators.registry import ValidatorRegistry
from datasetgen.validators.semantic import SemanticJudgeValidator


@pytest.fixture
def base_blueprint():
    return Blueprint(
        dataset_name="phase2_test",
        number_of_examples=10,
        domain="computer_science",
        topics={"general": 1.0},
        difficulty_distribution={"beginner": 1.0},
        fields=[BlueprintField(name="input", type="string")],
    )


# 1. Classification Validator Tests
def test_classification_validator(base_blueprint):
    val = ClassificationValidator()
    bp = Blueprint(
        dataset_name="sentiment_ds",
        number_of_examples=5,
        domain="sentiment",
        topics={"positive": 0.5, "negative": 0.5},
        difficulty_distribution={"beginner": 1.0},
        fields=[
            BlueprintField(name="text", type="string"),
            BlueprintField(name="label", type="enum", enum_values=["positive", "negative"]),
        ],
    )

    # Valid class
    ok_ex = {"text": "I love this product!", "label": "positive"}
    res_ok = val.validate(ok_ex, bp)
    assert res_ok.is_accepted is True

    # Invalid class
    bad_ex = {"text": "I love this product!", "label": "super_happy"}
    res_bad = val.validate(bad_ex, bp)
    assert res_bad.is_accepted is False
    assert "not in allowed classes" in res_bad.errors[0]

    # Missing text
    missing_text = {"text": "", "label": "positive"}
    res_missing = val.validate(missing_text, bp)
    assert res_missing.is_accepted is False


# 2. Code Sandbox Validator Tests
def test_code_sandbox_success(base_blueprint):
    val = CodeSandboxValidator(timeout_seconds=2.0)
    good_code = {
        "code": "def reverse_string(s: str) -> str:\n    return s[::-1]",
        "test_cases": "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''",
    }
    res = val.validate(good_code, base_blueprint)
    assert res.is_accepted is True


def test_code_sandbox_assertion_failure(base_blueprint):
    val = CodeSandboxValidator(timeout_seconds=2.0)
    buggy_code = {
        "code": "def add(a, b):\n    return a - b", # Buggy
        "test_cases": "assert add(2, 3) == 5",
    }
    res = val.validate(buggy_code, base_blueprint)
    assert res.is_accepted is False
    assert "Runtime assertion failure" in res.errors[0]


def test_code_sandbox_security_rejection(base_blueprint):
    val = CodeSandboxValidator(timeout_seconds=2.0)
    malicious_code = {
        "code": "import os\nos.system('echo dangerous')",
        "test_cases": "",
    }
    res = val.validate(malicious_code, base_blueprint)
    assert res.is_accepted is False
    assert any("Disallowed import of module 'os'" in err for err in res.errors)


def test_code_sandbox_syntax_error(base_blueprint):
    val = CodeSandboxValidator(timeout_seconds=2.0)
    syntax_error = {
        "code": "def invalid_syntax(a, b\n    return a",
        "test_cases": "",
    }
    res = val.validate(syntax_error, base_blueprint)
    assert res.is_accepted is False
    assert "syntax error" in res.errors[0].lower()


# 3. Context-QA Validator Tests
def test_context_qa_validator(base_blueprint):
    val = ContextQAValidator(min_support_ratio=0.5)

    # Grounded answer
    grounded = {
        "context": "Photosynthesis in green plants converts water and carbon dioxide into oxygen and glucose using sunlight.",
        "question": "What is produced during photosynthesis?",
        "answer": "Oxygen and glucose are produced.",
    }
    res_ok = val.validate(grounded, base_blueprint)
    assert res_ok.is_accepted is True

    # Hallucinated answer with unsupported claims
    hallucinated = {
        "context": "Photosynthesis in green plants converts water and carbon dioxide into oxygen and glucose.",
        "question": "What is produced?",
        "answer": "Plutonium, uranium, and radioactive elements are produced in large reactors.",
    }
    res_bad = val.validate(hallucinated, base_blueprint)
    assert res_bad.is_accepted is False
    assert "not supported by context" in res_bad.errors[0]


# 4. JSON Extraction Validator Tests
def test_json_extraction_validator(base_blueprint):
    val = JSONExtractionValidator()

    # Valid dict / json string
    valid_ex = {"extracted_json": '{"person": "Alice", "age": 30, "city": "Seattle"}'}
    res_ok = val.validate(valid_ex, base_blueprint)
    assert res_ok.is_accepted is True

    # Malformed json
    bad_json = {"extracted_json": '{"person": "Alice", age: }'}
    res_bad = val.validate(bad_json, base_blueprint)
    assert res_bad.is_accepted is False
    assert "Failed to parse extracted JSON" in res_bad.errors[0]

    # Empty json
    empty_json = {"extracted_json": "{}"}
    res_empty = val.validate(empty_json, base_blueprint)
    assert res_empty.is_accepted is False


# 5. Semantic Judge Validator Tests
def test_semantic_judge_repetition_check(base_blueprint):
    val = SemanticJudgeValidator()

    # Normal text
    good_text = {"instruction": "Explain gravity", "response": "Gravity is the fundamental force of attraction between masses."}
    res_ok = val.validate(good_text, base_blueprint)
    assert res_ok.is_accepted is True

    # Word repetition loop
    loop_text = {"instruction": "Explain gravity", "response": "gravity gravity gravity gravity gravity gravity gravity gravity"}
    res_loop = val.validate(loop_text, base_blueprint)
    assert res_loop.is_accepted is False
    assert "repetition loop" in res_loop.errors[0]


# 6. Registry Validation
def test_validator_registry_instantiates_phase2():
    for name in ["classification", "code_sandbox", "context_qa", "json_extraction", "semantic_judge"]:
        val = ValidatorRegistry.get_validator(name)
        assert val is not None
        assert val.name == name
