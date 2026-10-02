"""Evaluation metrics calculator for benchmark runs."""

from __future__ import annotations

import math
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from datasetgen.pipeline.orchestrator import GenerationReport
from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.trace import Trace


class BenchmarkMetrics(BaseModel):
    """Calculated metrics for an evaluated dataset generation run."""

    benchmark_id: str
    dataset_name: str
    total_requested: int
    total_generated: int
    total_accepted: int
    total_rejected: int

    # Quality Rates (0.0 to 1.0)
    acceptance_rate: float = 0.0
    schema_valid_rate: float = 0.0
    domain_valid_rate: float = 0.0
    constraint_satisfaction_rate: float = 0.0
    duplicate_rate: float = 0.0

    # Distribution Accuracy
    topic_distribution_error: float = 0.0
    difficulty_distribution_error: float = 0.0

    # Operational Efficiency
    regeneration_rate: float = 0.0
    tokens_per_accepted_example: float = 0.0
    latency_seconds: float = 0.0
    cost_per_1k_accepted_usd: float = 0.0

    # Quality Gate
    passed_quality_gate: bool = True
    gate_failures: List[str] = Field(default_factory=list)


def compute_benchmark_metrics(
    benchmark_id: str,
    blueprint: Blueprint,
    report: GenerationReport,
    trace: Trace,
) -> BenchmarkMetrics:
    """Compute rigorous benchmark metrics from pipeline execution telemetry."""
    total_gen = max(report.generated, 1)
    total_acc = report.accepted

    schema_rate = 1.0 - (report.schema_failures / total_gen)
    domain_rate = 1.0 - (report.domain_failures / total_gen)
    constraint_rate = 1.0 - (report.constraint_failures / total_gen)
    dup_rate = report.duplicates / total_gen

    # Calculate topic distribution error (Total Variation Distance)
    achieved_topics: Dict[str, int] = {}
    achieved_diffs: Dict[str, int] = {}
    for ex in trace.final_examples:
        achieved_topics[ex.topic] = achieved_topics.get(ex.topic, 0) + 1
        achieved_diffs[ex.difficulty] = achieved_diffs.get(ex.difficulty, 0) + 1

    t_err = 0.0
    if total_acc > 0:
        for t, target_frac in blueprint.topics.items():
            actual_frac = achieved_topics.get(t, 0) / total_acc
            t_err += abs(actual_frac - target_frac)
        t_err = t_err / 2.0  # TVD range [0, 1]

    d_err = 0.0
    if total_acc > 0:
        for d, target_frac in blueprint.difficulty_distribution.items():
            actual_frac = achieved_diffs.get(d, 0) / total_acc
            d_err += abs(actual_frac - target_frac)
        d_err = d_err / 2.0

    tokens_per_acc = (report.tokens_used / total_acc) if total_acc > 0 else 0.0
    cost_per_1k = (report.estimated_cost / total_acc * 1000) if total_acc > 0 else 0.0
    regen_rate = (report.regeneration_attempts / max(total_gen // 10, 1))

    # Evaluate Quality Gates
    gate_failures = []
    if schema_rate < 0.85:
        gate_failures.append(f"Schema validity {schema_rate:.2f} < 0.85")
    if domain_rate < 0.80:
        gate_failures.append(f"Domain validity {domain_rate:.2f} < 0.80")
    if dup_rate > 0.20:
        gate_failures.append(f"Duplicate rate {dup_rate:.2f} > 0.20")
    if report.acceptance_rate < 60.0:
        gate_failures.append(f"Acceptance rate {report.acceptance_rate:.1f}% < 60.0%")

    return BenchmarkMetrics(
        benchmark_id=benchmark_id,
        dataset_name=blueprint.dataset_name,
        total_requested=report.requested,
        total_generated=report.generated,
        total_accepted=report.accepted,
        total_rejected=report.rejected,
        acceptance_rate=round(report.acceptance_rate / 100.0, 4),
        schema_valid_rate=round(schema_rate, 4),
        domain_valid_rate=round(domain_rate, 4),
        constraint_satisfaction_rate=round(constraint_rate, 4),
        duplicate_rate=round(dup_rate, 4),
        topic_distribution_error=round(t_err, 4),
        difficulty_distribution_error=round(d_err, 4),
        regeneration_rate=round(regen_rate, 4),
        tokens_per_accepted_example=round(tokens_per_acc, 1),
        latency_seconds=round(report.duration_seconds, 2),
        cost_per_1k_accepted_usd=round(cost_per_1k, 4),
        passed_quality_gate=len(gate_failures) == 0,
        gate_failures=gate_failures,
    )
