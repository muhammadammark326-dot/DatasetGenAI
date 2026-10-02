"""Automated benchmark runner evaluating pipeline performance across categories."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional

from datasetgen.config.settings import get_settings
from datasetgen.evaluation.metrics import BenchmarkMetrics, compute_benchmark_metrics
from datasetgen.evaluation.report import BenchmarkReporter
from datasetgen.pipeline.orchestrator import DatasetPipeline
from datasetgen.providers.base import LLMProvider
from datasetgen.storage.jsonl_io import read_jsonl


class BenchmarkRunner:
    """Runs fixed benchmark requests to track quality regressions and model performance."""

    def __init__(
        self,
        provider: LLMProvider,
        benchmark_file: Optional[Path] = None,
        reporter: Optional[BenchmarkReporter] = None,
    ) -> None:
        self.provider = provider
        self.benchmark_file = benchmark_file or (Path("benchmarks") / "starter_benchmarks.jsonl")
        self.reporter = reporter or BenchmarkReporter()
        self.pipeline = DatasetPipeline(provider=provider, batch_size=10)

    def run_all(self, limit: Optional[int] = None) -> List[BenchmarkMetrics]:
        """Execute all benchmark requests and export reports."""
        records = read_jsonl(self.benchmark_file)
        if limit:
            records = records[:limit]

        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        all_metrics: List[BenchmarkMetrics] = []

        for record in records:
            b_id = record.get("id", "bench_unknown")
            req = record.get("request", "")

            # Plan and run
            blueprint = self.pipeline.plan(req)
            accepted, report, trace = self.pipeline.run(
                user_request=req,
                blueprint=blueprint,
                approval_callback=lambda bp: True,
            )

            metrics = compute_benchmark_metrics(
                benchmark_id=b_id,
                blueprint=blueprint,
                report=report,
                trace=trace,
            )
            all_metrics.append(metrics)

        # Generate combined report
        self.reporter.generate_report(run_id=run_id, results=all_metrics)
        return all_metrics
