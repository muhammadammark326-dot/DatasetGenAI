"""Extracts direct preference optimization (DPO) pairs from decision traces."""

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


class DPODataPreparer:
    """Transforms accepted vs rejected generation traces into leak-free DPO pairs."""

    def __init__(
        self,
        trace_store: Optional[TraceStore] = None,
        output_dir: Optional[Path] = None,
        benchmark_file: Optional[Path] = None,
    ) -> None:
        self.trace_store = trace_store or TraceStore()
        settings = get_settings()
        self.output_dir = output_dir or (settings.data_dir / "dpo_data")
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
        """Extract (chosen, rejected) preference pairs from traces and export train/val splits."""
        traces = self.trace_store.list_traces()
        rng = random.Random(seed)

        dpo_pairs: List[Dict[str, Any]] = []

        for trace in traces:
            # Prevent benchmark leakage
            if self.is_leaked_request(trace.user_request):
                continue

            if not trace.final_examples or not trace.rejected_examples:
                continue

            # Group accepted examples by topic
            topic_to_accepted: Dict[str, List[Any]] = {}
            for acc in trace.final_examples:
                topic_to_accepted.setdefault(acc.topic, []).append(acc)

            for rej in trace.rejected_examples:
                candidates = topic_to_accepted.get(rej.topic, trace.final_examples)
                if not candidates:
                    continue

                chosen_example = candidates[0]
                errors = [e for v in rej.validation_history for e in v.errors] if rej.validation_history else []

                prompt = (
                    f"Generate a validated, high-quality dataset example for domain '{trace.blueprint.domain}', "
                    f"topic '{rej.topic}', difficulty '{rej.difficulty}'. "
                    f"Format: {trace.blueprint.output_format}."
                )

                dpo_pairs.append({
                    "prompt": prompt,
                    "chosen": json.dumps(chosen_example.data, indent=2),
                    "rejected": json.dumps(rej.data, indent=2),
                    "metadata": {
                        "trace_id": trace.trace_id,
                        "topic": rej.topic,
                        "difficulty": rej.difficulty,
                        "rejection_errors": errors,
                    },
                })

        # Shuffle and split
        rng.shuffle(dpo_pairs)
        split_idx = int(len(dpo_pairs) * train_ratio)
        train_data = dpo_pairs[:split_idx]
        val_data = dpo_pairs[split_idx:]

        train_path = self.output_dir / "train_dpo.jsonl"
        val_path = self.output_dir / "val_dpo.jsonl"
        meta_path = self.output_dir / "dpo_metadata.json"

        write_jsonl(train_path, train_data)
        write_jsonl(val_path, val_data)

        summary = {
            "total_pairs": len(dpo_pairs),
            "train_count": len(train_data),
            "val_count": len(val_data),
            "train_ratio": train_ratio,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return train_path, val_path, summary
