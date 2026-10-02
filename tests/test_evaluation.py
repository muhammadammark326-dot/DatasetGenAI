"""Unit tests for evaluation metrics, reporting, and benchmark execution."""

import tempfile
from pathlib import Path

from datasetgen.evaluation.benchmark_runner import BenchmarkRunner
from datasetgen.evaluation.metrics import BenchmarkMetrics, compute_benchmark_metrics
from datasetgen.evaluation.report import BenchmarkReporter
from datasetgen.pipeline.orchestrator import GenerationReport
from datasetgen.providers.mock import MockProvider
from datasetgen.schemas.blueprint import Blueprint, BlueprintField
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.trace import Trace
from datasetgen.schemas.validation import ValidationResult
from datasetgen.storage.jsonl_io import write_jsonl


def test_compute_metrics():
    bp = Blueprint(
        dataset_name="metrics_test",
        number_of_examples=10,
        domain="physics",
        topics={"mechanics": 0.5, "electricity": 0.5},
        difficulty_distribution={"beginner": 0.5, "advanced": 0.5},
        fields=[BlueprintField(name="q", type="string")],
    )

    report = GenerationReport()
    report.requested = 10
    report.generated = 12
    report.accepted = 10
    report.rejected = 2
    report.schema_failures = 1
    report.duplicates = 1
    report.domain_failures = 0
    report.duration_seconds = 1.2
    report.tokens_used = 500

    trace = Trace(
        user_request="test",
        blueprint=bp,
        generator_model="mock",
        final_examples=[
            GeneratedExample(data={"q": "1"}, topic="mechanics", difficulty="beginner"),
            GeneratedExample(data={"q": "2"}, topic="electricity", difficulty="advanced"),
        ],
    )

    metrics = compute_benchmark_metrics(
        benchmark_id="bench_test_01",
        blueprint=bp,
        report=report,
        trace=trace,
    )

    assert metrics.benchmark_id == "bench_test_01"
    assert metrics.total_accepted == 10
    assert metrics.schema_valid_rate > 0.90
    assert metrics.passed_quality_gate is True


def test_benchmark_runner_and_reporter():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bench_file = tmp_path / "test_bench.jsonl"
        write_jsonl(bench_file, [
            {"id": "test_01", "request": "Generate 5 high-school physics questions on mechanics."}
        ])

        reporter = BenchmarkReporter(output_dir=tmp_path / "reports")
        provider = MockProvider(seed=42, error_rate=0.0)
        runner = BenchmarkRunner(provider=provider, benchmark_file=bench_file, reporter=reporter)

        results = runner.run_all()
        assert len(results) == 1
        assert results[0].benchmark_id == "test_01"
        assert results[0].total_accepted == 5

        # Verify output files
        assert len(list((tmp_path / "reports").glob("*.md"))) == 1
        assert len(list((tmp_path / "reports").glob("*.json"))) == 1
