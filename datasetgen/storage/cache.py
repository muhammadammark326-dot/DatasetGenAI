"""Cache engine for batch generation responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from datasetgen.config.settings import get_settings


class GenerationCache:
    """Disk-backed cache keyed by generation parameters."""

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True) -> None:
        settings = get_settings()
        self.cache_dir = cache_dir or settings.cache_dir
        self.enabled = enabled and settings.cache_enabled
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(
        blueprint_dict: Dict[str, Any],
        model: str,
        prompt_version: str,
        batch_index: int,
        seed: int,
    ) -> str:
        """Create a deterministic hash key from generation arguments."""
        payload = {
            "blueprint": blueprint_dict,
            "model": model,
            "prompt_version": prompt_version,
            "batch_index": batch_index,
            "seed": seed,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached generation response if available."""
        if not self.enabled:
            return None
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store generation response in cache."""
        if not self.enabled:
            return
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
