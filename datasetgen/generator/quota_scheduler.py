"""Quota scheduler balancing topic and difficulty distribution buckets."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from datasetgen.schemas.blueprint import Blueprint


class QuotaScheduler:
    """Calculates and monitors target vs actual counts across topic x difficulty buckets."""

    def __init__(self, blueprint: Blueprint) -> None:
        self.blueprint = blueprint
        self.total_target = blueprint.number_of_examples
        self.target_quotas: Dict[Tuple[str, str], int] = self._calculate_quotas()
        self.accepted_counts: Dict[Tuple[str, str], int] = {
            bucket: 0 for bucket in self.target_quotas
        }

    def _calculate_quotas(self) -> Dict[Tuple[str, str], int]:
        quotas: Dict[Tuple[str, str], int] = {}
        allocated = 0
        buckets = []

        for topic, t_frac in self.blueprint.topics.items():
            for diff, d_frac in self.blueprint.difficulty_distribution.items():
                target = int(round(self.total_target * t_frac * d_frac))
                quotas[(topic, diff)] = target
                allocated += target
                buckets.append((topic, diff))

        # Adjust any rounding discrepancy on the largest bucket
        diff = self.total_target - allocated
        if diff != 0 and buckets:
            quotas[buckets[0]] += diff

        return quotas

    def record_acceptance(self, topic: str, difficulty: str) -> None:
        """Increment count for accepted example."""
        key = (topic, difficulty)
        if key in self.accepted_counts:
            self.accepted_counts[key] += 1
        else:
            self.accepted_counts[key] = 1

    def is_complete(self) -> bool:
        """Check if all buckets have reached their target quota."""
        total_accepted = sum(self.accepted_counts.values())
        return total_accepted >= self.total_target

    def get_remaining_needed(self) -> int:
        return max(0, self.total_target - sum(self.accepted_counts.values()))

    def get_next_target_bucket(self) -> Tuple[str, str, int]:
        """Return the most under-filled (topic, difficulty) bucket and count needed."""
        deficits = [
            (
                bucket,
                self.target_quotas[bucket] - self.accepted_counts.get(bucket, 0),
            )
            for bucket in self.target_quotas
        ]
        # Sort by largest deficit first
        deficits.sort(key=lambda x: x[1], reverse=True)

        bucket, deficit = deficits[0]
        # Return bucket and minimum of deficit or batch size
        return bucket[0], bucket[1], max(1, deficit)

    def get_distribution_stats(self) -> Dict[str, Any]:
        """Return summary of target vs accepted per bucket."""
        stats = {}
        for (topic, diff), target in self.target_quotas.items():
            actual = self.accepted_counts.get((topic, diff), 0)
            stats[f"{topic}:{diff}"] = {"target": target, "actual": actual}
        return stats
