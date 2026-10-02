"""Pipeline orchestrator for end-to-end dataset generation and validation."""

from __future__ import annotations

import datetime
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasetgen.config.settings import get_settings
from datasetgen.generator.batcher import BatchGenerator
from datasetgen.generator.quota_scheduler import QuotaScheduler
from datasetgen.planner.blueprint_builder import BlueprintBuilder
from datasetgen.providers.base import LLMProvider
from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.trace import Trace
from datasetgen.schemas.validation import ValidationResult
from datasetgen.storage.checkpoint import CheckpointManager, CheckpointState
from datasetgen.storage.jsonl_io import write_jsonl
from datasetgen.storage.trace_store import TraceStore
from datasetgen.storage.versioning import DatasetProvenance
from datasetgen.validators.registry import ValidatorRegistry


class GenerationReport:
    """Statistical summary of generation and validation run."""

    def __init__(self) -> None:
        self.requested: int = 0
        self.generated: int = 0
        self.accepted: int = 0
        self.rejected: int = 0
        self.duplicates: int = 0
        self.domain_failures: int = 0
        self.schema_failures: int = 0
        self.constraint_failures: int = 0
        self.semantic_failures: int = 0
        self.regeneration_attempts: int = 0
        self.tokens_used: int = 0
        self.estimated_cost: float = 0.0
        self.duration_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted / self.generated * 100) if self.generated > 0 else 0.0

    def format_cli(self) -> str:
        lines = [
            "=" * 68,
            "                   DATASETGEN AI EXECUTION REPORT                   ",
            "=" * 68,
            f"Requested: {self.requested:<8} Generated: {self.generated:<8} Accepted: {self.accepted:<8} Rejected: {self.rejected:<8}",
            f"Duplicates: {self.duplicates:<7} Domain failures: {self.domain_failures:<4} Schema failures: {self.schema_failures:<4} Semantic failures: {self.semantic_failures:<4}",
            f"Regeneration attempts: {self.regeneration_attempts:<4} Acceptance rate: {self.acceptance_rate:.1f}%",
            f"Tokens: {self.tokens_used:<11} Cost: ${self.estimated_cost:.4f} Duration: {self.duration_seconds:.2f}s",
            "=" * 68,
        ]
        return "\n".join(lines)


class DatasetPipeline:
    """Orchestrates Planner -> Blueprint approval -> Batch generation -> Multi-stage validation -> Regeneration -> Final export."""

    def __init__(
        self,
        provider: LLMProvider,
        checkpoint_manager: Optional[CheckpointManager] = None,
        trace_store: Optional[TraceStore] = None,
        batch_size: int = 20,
    ) -> None:
        self.provider = provider
        self.settings = get_settings()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.trace_store = trace_store or TraceStore()
        self.batch_size = batch_size
        self.batch_generator = BatchGenerator(provider)

    def plan(self, user_request: str, overrides: Optional[Dict[str, Any]] = None) -> Blueprint:
        """Create a blueprint from user request."""
        builder = BlueprintBuilder(self.provider)
        return builder.build_blueprint(user_request, overrides=overrides)

    def run(
        self,
        user_request: str,
        blueprint: Optional[Blueprint] = None,
        approval_callback: Optional[Callable[[Blueprint], bool]] = None,
        run_id: Optional[str] = None,
        resume: bool = True,
    ) -> Tuple[List[GeneratedExample], GenerationReport, Trace]:
        """Execute full end-to-end dataset generation."""
        start_time = time.time()
        report = GenerationReport()

        # 1. Plan if blueprint not supplied
        if blueprint is None:
            blueprint = self.plan(user_request)

        report.requested = blueprint.number_of_examples

        # 2. User Approval
        if approval_callback is not None:
            approved = approval_callback(blueprint)
            if not approved:
                raise RuntimeError("Blueprint was rejected by user. Aborting generation.")

        # 3. Setup validators
        validators = ValidatorRegistry.build_chain(blueprint)
        scheduler = QuotaScheduler(blueprint)

        # Checkpoint restoration if available
        cid = run_id or f"{blueprint.dataset_name}_{blueprint.random_seed}"
        accepted_examples: List[GeneratedExample] = []
        rejected_examples: List[GeneratedExample] = []
        all_validation_results: List[ValidationResult] = []
        regeneration_attempts: List[Dict[str, Any]] = []

        if resume:
            state = self.checkpoint_manager.load(cid)
            if state:
                accepted_examples = state.accepted_examples
                rejected_examples = state.rejected_examples
                for ex in accepted_examples:
                    scheduler.record_acceptance(ex.topic, ex.difficulty)
                report.accepted = len(accepted_examples)
                report.rejected = len(rejected_examples)
                report.generated = report.accepted + report.rejected

        batch_index = 0
        consecutive_failures = 0
        max_retries = blueprint.max_retries or 3

        # 4. Main Generation & Targeted Regeneration Loop
        while not scheduler.is_complete():
            topic, difficulty, needed = scheduler.get_next_target_bucket()
            current_batch_count = min(self.batch_size, needed)

            # Generate batch
            batch_examples, llm_resp = self.batch_generator.generate_batch(
                blueprint=blueprint,
                topic=topic,
                difficulty=difficulty,
                count=current_batch_count,
                batch_index=batch_index,
            )

            report.tokens_used += llm_resp.usage.total_tokens
            report.estimated_cost += llm_resp.usage.estimated_cost_usd
            batch_index += 1

            batch_accepted = 0

            for ex in batch_examples:
                report.generated += 1
                is_valid = True

                # Run through validator chain
                for val in validators:
                    res = val.validate(ex, blueprint)
                    ex.add_validation(res)
                    all_validation_results.append(res)

                    if not res.is_accepted:
                        is_valid = False
                        # Track failure categories for telemetry report
                        if "dedup" in res.validator:
                            report.duplicates += 1
                        elif "physics" in res.validator or "math" in res.validator:
                            report.domain_failures += 1
                        elif "schema" in res.validator:
                            report.schema_failures += 1
                        elif "constraint" in res.validator:
                            report.constraint_failures += 1
                        else:
                            report.semantic_failures += 1
                        break

                if is_valid:
                    accepted_examples.append(ex)
                    scheduler.record_acceptance(ex.topic, ex.difficulty)
                    report.accepted += 1
                    batch_accepted += 1
                else:
                    rejected_examples.append(ex)
                    report.rejected += 1

            if batch_accepted == 0:
                consecutive_failures += 1
                report.regeneration_attempts += 1
                regeneration_attempts.append({
                    "topic": topic,
                    "difficulty": difficulty,
                    "attempt": consecutive_failures,
                    "time": time.time(),
                })
                # Exponential backoff on consecutive batch failures
                if consecutive_failures > max_retries:
                    # Switch target bucket or log warning
                    consecutive_failures = 0
            else:
                consecutive_failures = 0

            # Save checkpoint
            chk_state = CheckpointState(
                run_id=cid,
                blueprint=blueprint,
                completed_batches=batch_index,
                accepted_examples=accepted_examples,
                rejected_examples=rejected_examples,
                tokens_used=report.tokens_used,
                estimated_cost=report.estimated_cost,
            )
            self.checkpoint_manager.save(chk_state)

            # Safety breakout if generation gets stuck
            if report.generated > blueprint.number_of_examples * 10:
                break

        report.duration_seconds = time.time() - start_time

        # 5. Export Datasets & Provenance
        ds_dir = self.settings.datasets_dir / blueprint.dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        final_data_records = [ex.data for ex in accepted_examples]
        final_jsonl_path = ds_dir / "final.jsonl"
        write_jsonl(final_jsonl_path, final_data_records)

        # Save approved Blueprint
        bp_path = ds_dir / "blueprint.json"
        with open(bp_path, "w", encoding="utf-8") as f:
            f.write(blueprint.model_dump_json(indent=2))

        # Save Provenance
        provenance = DatasetProvenance(
            dataset_name=blueprint.dataset_name,
            dataset_version=blueprint.dataset_version,
            blueprint_version=blueprint.blueprint_version,
            generator_model=blueprint.generator_model,
            validator_versions={v.identifier: v.version for v in validators},
            random_seed=blueprint.random_seed,
            total_accepted=len(accepted_examples),
            total_generated=report.generated,
        )
        provenance.save_to_dir(ds_dir)

        # 6. Save Decision Trace
        trace = Trace(
            user_request=user_request,
            blueprint=blueprint,
            blueprint_version=blueprint.blueprint_version,
            generator_model=blueprint.generator_model,
            prompt_version=blueprint.prompt_version,
            examples=accepted_examples + rejected_examples,
            validation_results=all_validation_results,
            quality_scores={"acceptance_rate": report.acceptance_rate},
            rejected_examples=rejected_examples,
            regeneration_attempts=regeneration_attempts,
            final_examples=accepted_examples,
        )
        self.trace_store.save_trace(trace)

        # Clean checkpoint on completion
        self.checkpoint_manager.remove(cid)

        return accepted_examples, report, trace
