"""Unit tests for SFT dataset preparation and leakage prevention."""

import tempfile
from pathlib import Path

from datasetgen.schemas.blueprint import Blueprint, BlueprintField
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.trace import Trace
from datasetgen.schemas.validation import ValidationResult
from datasetgen.storage.jsonl_io import read_jsonl, write_jsonl
from datasetgen.storage.trace_store import TraceStore
from datasetgen.training.prepare_sft import SFTDataPreparer


def test_sft_preparation_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trace_store = TraceStore(traces_dir=tmp_path / "traces")
        output_dir = tmp_path / "sft_out"
        bench_file = tmp_path / "benchmarks.jsonl"

        # Create a benchmark file with a protected query
        write_jsonl(bench_file, [
            {"id": "b1", "request": "Protected benchmark query for physics"}
        ])

        bp = Blueprint(
            dataset_name="sft_test_ds",
            number_of_examples=5,
            domain="physics",
            topics={"mechanics": 1.0},
            difficulty_distribution={"beginner": 1.0},
            fields=[BlueprintField(name="q", type="string"), BlueprintField(name="a", type="string")],
        )

        # 1. Normal trace (should be included)
        t_ok = Trace(
            user_request="Normal request for 5 physics questions",
            blueprint=bp,
            generator_model="mock",
            final_examples=[
                GeneratedExample(
                    data={"q": "Question 1", "a": "Answer 1"},
                    topic="mechanics",
                    difficulty="beginner",
                    is_valid=True,
                )
            ],
            rejected_examples=[
                GeneratedExample(
                    data={"q": "Bad Q", "a": "Bad A"},
                    topic="mechanics",
                    difficulty="beginner",
                    validation_history=[
                        ValidationResult(status="reject", validator="test_val", errors=["Invalid calculation"])
                    ],
                )
            ],
            regeneration_attempts=[
                {"topic": "mechanics", "difficulty": "beginner", "attempt": 1}
            ],
        )
        trace_store.save_trace(t_ok)

        # 2. Leaked trace (should be excluded)
        t_leaked = Trace(
            user_request="Protected benchmark query for physics",
            blueprint=bp,
            generator_model="mock",
            final_examples=[GeneratedExample(data={"q": "Leaked"}, topic="mechanics", difficulty="beginner")],
        )
        trace_store.save_trace(t_leaked)

        preparer = SFTDataPreparer(
            trace_store=trace_store,
            output_dir=output_dir,
            benchmark_file=bench_file,
        )

        train_p, val_p, summary = preparer.prepare_dataset(train_ratio=0.5, seed=42)

        assert train_p.exists()
        assert val_p.exists()
        assert summary["total_records"] > 0

        # Verify benchmark leakage exclusion
        train_records = read_jsonl(train_p) + read_jsonl(val_p)
        for r in train_records:
            assert "Protected benchmark query" not in r["instruction"]

        # Verify task categories present
        tasks = {r["task"] for r in train_records}
        assert "request_to_blueprint" in tasks
        assert "blueprint_to_batch" in tasks
        assert "error_repair" in tasks
        assert "targeted_generation" in tasks
