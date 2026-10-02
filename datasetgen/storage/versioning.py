"""Dataset provenance and versioning metadata."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetProvenance(BaseModel):
    """Provenance record documenting the origin and validation pedigree of a dataset."""

    dataset_name: str
    dataset_version: str = "1.0.0"
    blueprint_version: str = "1"
    generator_model: str
    validator_versions: Dict[str, str] = Field(default_factory=dict)
    source_datasets: List[str] = Field(default_factory=list)
    license: str = "MIT"
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    random_seed: int = 42
    total_accepted: int = 0
    total_generated: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def save_to_dir(self, directory: Path) -> Path:
        """Write provenance.json in the dataset folder."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "provenance.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return path
