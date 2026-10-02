"""Markdown and JSON report generator for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from datasetgen.config.settings import get_settings
from datasetgen.evaluation.metrics import BenchmarkMetrics


class BenchmarkReporter:
    """Generates visual Markdown and JSON benchmark summaries."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or get_settings().benchmark_results_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, run_id: str, results: List[BenchmarkMetrics]) -> Path:
        """Create JSON and Markdown reports."""
        json_path = self.output_dir / f"benchmark_report_{run_id}.json"
        md_path = self.output_dir / f"benchmark_report_{run_id}.md"

        # 1. Export JSON
        data = [r.model_dump(mode="json") for r in results]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "results": data}, f, indent=2)

        # 2. Export Markdown
        md_lines = [
            f"# DatasetGen AI — Benchmark Evaluation Report",
            f"**Run ID:** `{run_id}`  ",
            f"**Evaluated Datasets:** {len(results)}  \n",
            "| Benchmark ID | Dataset | Accepted | Generated | Acceptance Rate | Schema Valid | Domain Valid | Duplicates | Gate Status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        for r in results:
            status_icon = "PASSED" if r.passed_quality_gate else "FAILED"
            md_lines.append(
                f"| `{r.benchmark_id}` | {r.dataset_name} | {r.total_accepted} | {r.total_generated} | {r.acceptance_rate*100:.1f}% | {r.schema_valid_rate*100:.1f}% | {r.domain_valid_rate*100:.1f}% | {r.duplicate_rate*100:.1f}% | **{status_icon}** |"
            )

        md_lines.append("\n## Operational Telemetry\n")
        md_lines.append("| Benchmark ID | Tokens / Accepted | Latency (s) | Cost / 1k ($) | Topic Dist Error | Diff Dist Error |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in results:
            md_lines.append(
                f"| `{r.benchmark_id}` | {r.tokens_per_accepted_example:.1f} | {r.latency_seconds:.2f}s | ${r.cost_per_1k_accepted_usd:.4f} | {r.topic_distribution_error:.3f} | {r.difficulty_distribution_error:.3f} |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return md_path
