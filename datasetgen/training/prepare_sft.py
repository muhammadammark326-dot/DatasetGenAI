"""Extracts and formats decision traces into supervised fine-tuning (SFT) datasets."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from datasetgen.config.settings import get_settings
from datasetgen.schemas.trace import Trace
from datasetgen.storage.jsonl_io import read_jsonl, write_jsonl
from datasetgen.storage.trace_store import TraceStore


class SFTDataPreparer:
    """Transforms raw pipeline traces into leak-free SFT training datasets."""

    def __init__(
        self,
        trace_store: Optional[TraceStore] = None,
        output_dir: Optional[Path] = None,
        benchmark_file: Optional[Path] = None,
    ) -> None:
        self.trace_store = trace_store or TraceStore()
        settings = get_settings()
        self.output_dir = output_dir or settings.sft_data_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_file = benchmark_file or (Path("benchmarks") / "starter_benchmarks.jsonl")
        self.benchmark_hashes: Set[str] = self._load_benchmark_hashes()

    def _load_benchmark_hashes(self) -> Set[str]:
        hashes = set()
        if self.benchmark_file.exists():
            records = read_jsonl(self.benchmark_file)
            for r in records:
                req = r.get("request", "").strip().lower()
                if req:
                    hashes.add(hashlib.sha256(req.encode("utf-8")).hexdigest())
        return hashes

    def is_leaked_request(self, user_request: str) -> bool:
        """Check if request overlaps with the evaluation benchmark."""
        clean = user_request.strip().lower()
        req_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        return req_hash in self.benchmark_hashes

    def prepare_dataset(
        self,
        train_ratio: float = 0.90,
        seed: int = 42,
    ) -> Tuple[Path, Path, Dict[str, Any]]:
        """Extract multi-task SFT pairs and export train/val splits."""
        traces = self.trace_store.list_traces()
        rng = random.Random(seed)

        sft_records: List[Dict[str, Any]] = []

        for trace in traces:
            # 1. Prevent benchmark leakage
            if self.is_leaked_request(trace.user_request):
                continue

            # Task 1: Request -> Blueprint
            sft_records.append({
                "task": "request_to_blueprint",
                "instruction": f"Convert this natural language request into a dataset blueprint:\nRequest: {trace.user_request}",
                "response": trace.blueprint.model_dump_json(indent=2),
                "metadata": {"trace_id": trace.trace_id},
            })

            # Task 2: Request + Blueprint -> Structured Batch
            if trace.final_examples:
                batch_sample = [ex.data for ex in trace.final_examples[:5]]
                sft_records.append({
                    "task": "blueprint_to_batch",
                    "instruction": f"Generate a structured batch for dataset '{trace.blueprint.dataset_name}' with domain '{trace.blueprint.domain}':\nBlueprint fields: {[f.name for f in trace.blueprint.fields]}",
                    "response": json.dumps({"examples": batch_sample}, indent=2),
                    "metadata": {"trace_id": trace.trace_id},
                })

            # Task 3: Repair (Failed Example + Validator Error -> Corrected Example)
            for rej in trace.rejected_examples:
                # Find matching accepted example in same topic if available
                matching_acc = next(
                    (ex for ex in trace.final_examples if ex.topic == rej.topic), None
                )
                if matching_acc and rej.validation_history:
                    errors = [e for v in rej.validation_history for e in v.errors]
                    sft_records.append({
                        "task": "error_repair",
                        "instruction": f"Repair this failed dataset example.\nFailed Example: {json.dumps(rej.data)}\nValidator Errors: {errors}",
                        "response": json.dumps(matching_acc.data, indent=2),
                        "metadata": {"trace_id": trace.trace_id, "errors": errors},
                    })

            # Task 4: Targeted Generation for Underfilled Buckets
            for attempt in trace.regeneration_attempts:
                topic = attempt.get("topic")
                diff = attempt.get("difficulty")
                matching = [
                    ex.data for ex in trace.final_examples
                    if ex.topic == topic and ex.difficulty == diff
                ][:3]
                if matching:
                    sft_records.append({
                        "task": "targeted_generation",
                        "instruction": f"Generate targeted examples for under-filled bucket topic='{topic}', difficulty='{diff}' in dataset '{trace.blueprint.dataset_name}'.",
                        "response": json.dumps({"examples": matching}, indent=2),
                        "metadata": {"trace_id": trace.trace_id, "topic": topic, "difficulty": diff},
                    })

        # Shuffle and split
        rng.shuffle(sft_records)
        split_idx = int(len(sft_records) * train_ratio)
        train_data = sft_records[:split_idx]
        val_data = sft_records[split_idx:]

        train_path = self.output_dir / "train_sft.jsonl"
        val_path = self.output_dir / "val_sft.jsonl"
        meta_path = self.output_dir / "sft_metadata.json"

        write_jsonl(train_path, train_data)
        write_jsonl(val_path, val_data)

        summary = {
            "total_records": len(sft_records),
            "train_count": len(train_data),
            "val_count": len(val_data),
            "train_ratio": train_ratio,
            "tasks": {
                "request_to_blueprint": len([r for r in sft_records if r["task"] == "request_to_blueprint"]),
                "blueprint_to_batch": len([r for r in sft_records if r["task"] == "blueprint_to_batch"]),
                "error_repair": len([r for r in sft_records if r["task"] == "error_repair"]),
                "targeted_generation": len([r for r in sft_records if r["task"] == "targeted_generation"]),
            },
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return train_path, val_path, summary
