"""Unit tests for storage, caching, and trace persistence."""

import tempfile
from pathlib import Path
from datasetgen.schemas.blueprint import Blueprint, BlueprintField
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.trace import Trace
from datasetgen.storage.cache import GenerationCache
from datasetgen.storage.checkpoint import CheckpointManager, CheckpointState
from datasetgen.storage.jsonl_io import append_jsonl, read_jsonl, write_jsonl
from datasetgen.storage.trace_store import TraceStore


def test_jsonl_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        items = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        write_jsonl(path, items)

        read_items = read_jsonl(path)
        assert len(read_items) == 2
        assert read_items[0]["text"] == "hello"

        append_jsonl(path, {"id": 3, "text": "extra"})
        assert len(read_jsonl(path)) == 3


def test_generation_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = GenerationCache(cache_dir=Path(tmpdir), enabled=True)
        key = cache.compute_key(
            blueprint_dict={"name": "test"},
            model="mock",
            prompt_version="v1",
            batch_index=0,
            seed=42,
        )

        assert cache.get(key) is None
        cache.set(key, {"data": "cached_example"})
        val = cache.get(key)
        assert val is not None
        assert val["data"] == "cached_example"


def test_trace_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TraceStore(traces_dir=Path(tmpdir))
        bp = Blueprint(
            dataset_name="trace_test_ds",
            number_of_examples=5,
            domain="physics",
            topics={"mechanics": 1.0},
            difficulty_distribution={"beginner": 1.0},
            fields=[BlueprintField(name="q", type="string")],
        )
        trace = Trace(
            user_request="give me 5 questions",
            blueprint=bp,
            generator_model="mock",
        )
        saved_path = store.save_trace(trace)
        assert saved_path.exists()

        loaded = store.load_trace(trace.trace_id)
        assert loaded is not None
        assert loaded.trace_id == trace.trace_id
