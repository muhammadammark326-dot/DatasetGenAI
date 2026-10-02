"""Unit tests for DPO preference data extraction and benchmark leakage checks."""

import tempfile
from pathlib import Path
from datasetgen.schemas.blueprint import Blueprint, BlueprintField
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.trace import Trace
from datasetgen.schemas.validation import ValidationResult
from datasetgen.storage.jsonl_io import write_jsonl
from datasetgen.storage.trace_store import TraceStore
from datasetgen.training.prepare_dpo import DPODataPreparer


def test_dpo_preparation_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trace_dir = tmp_path / "traces"
        output_dir = tmp_path / "dpo_out"
        bench_file = tmp_path / "benchmarks.jsonl"

        # Create a benchmark file with a protected query
        write_jsonl(bench_file, [
            {"id": "b1", "request": "Protected benchmark query for physics"}
        ])

        trace_store = TraceStore(traces_dir=trace_dir)

        bp = Blueprint(
            dataset_name="dpo_test_ds",
            number_of_examples=5,
            domain="physics",
            topics={"mechanics": 1.0},
            difficulty_distribution={"beginner": 1.0},
            fields=[BlueprintField(name="q", type="string"), BlueprintField(name="a", type="string")],
        )

        # 1. Normal trace (should produce a DPO pair)
        ex_accepted = GeneratedExample(
            data={"q": "Question 1", "a": "Answer 1"},
            topic="mechanics",
            difficulty="beginner",
            is_valid=True,
        )
        ex_rejected = GeneratedExample(
            data={"q": "Bad Q", "a": "Bad A"},
            topic="mechanics",
            difficulty="beginner",
            validation_history=[
                ValidationResult(status="reject", validator="test_val", errors=["Invalid calculation"])
            ],
        )

        t_ok = Trace(
            user_request="Normal request for 5 physics questions",
            blueprint=bp,
            generator_model="mock",
            final_examples=[ex_accepted],
            rejected_examples=[ex_rejected],
        )
        trace_store.save_trace(t_ok)

        # 2. Leaked Trace (matching benchmark request - must be excluded)
        t_leaked = Trace(
            user_request="Protected benchmark query for physics",
            blueprint=bp,
            generator_model="mock",
            final_examples=[ex_accepted],
            rejected_examples=[ex_rejected],
        )
        trace_store.save_trace(t_leaked)

        # Run DPO Preparer
        preparer = DPODataPreparer(
            trace_store=trace_store,
            output_dir=output_dir,
            benchmark_file=bench_file,
        )

        train_p, val_p, summary = preparer.prepare_dataset(train_ratio=0.8, seed=42)

        # Assertions
        assert summary["total_pairs"] == 1  # Leaked trace was excluded
        assert summary["train_count"] == 0 or summary["train_count"] == 1
        assert train_p.exists()
        assert val_p.exists()
