"""Tests for LLM provider registry and factory."""

import pytest
from datasetgen.providers import get_provider
from datasetgen.providers.mock import MockProvider
from datasetgen.providers.hf_local import HuggingFaceLocalProvider


def test_get_mock_provider():
    p = get_provider("mock")
    assert isinstance(p, MockProvider)
    res = p.generate("Hello")
    assert res.text != ""
    assert res.usage.total_tokens > 0


def test_get_hf_local_provider_instantiation():
    p = get_provider("hf_local", adapter_path="/tmp/fake_adapter")
    assert isinstance(p, HuggingFaceLocalProvider)
    assert p.adapter_path == "/tmp/fake_adapter"
    assert p.model_name == "Qwen/Qwen2.5-1.5B-Instruct"


def test_invalid_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("non_existent_provider_xyz")
