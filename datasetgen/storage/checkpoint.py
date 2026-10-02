"""Checkpoint and resume management for resilient execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from datasetgen.config.settings import get_settings
from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample


class CheckpointState(BaseModel):
    """Snapshot of in-progress dataset generation."""

    run_id: str
    blueprint: Blueprint
    completed_batches: int = 0
    accepted_examples: List[GeneratedExample] = Field(default_factory=list)
    rejected_examples: List[GeneratedExample] = Field(default_factory=list)
    bucket_counts: Dict[str, int] = Field(default_factory=dict)
    tokens_used: int = 0
    estimated_cost: float = 0.0
    status: str = "in_progress"


class CheckpointManager:
    """Handles writing and recovering generation checkpoints."""

    def __init__(self, checkpoints_dir: Optional[Path] = None) -> None:
        self.checkpoints_dir = checkpoints_dir or get_settings().checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, run_id: str) -> Path:
        return self.checkpoints_dir / f"checkpoint_{run_id}.json"

    def save(self, state: CheckpointState) -> Path:
        path = self.get_path(state.run_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
        return path

    def load(self, run_id: str) -> Optional[CheckpointState]:
        path = self.get_path(run_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CheckpointState.model_validate(data)

    def remove(self, run_id: str) -> None:
        path = self.get_path(run_id)
        if path.exists():
            path.unlink(missing_ok=True)
