"""Trace persistence engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from datasetgen.config.settings import get_settings
from datasetgen.schemas.trace import Trace
from datasetgen.storage.jsonl_io import append_jsonl, read_jsonl


class TraceStore:
    """Manages appending and retrieving decision traces."""

    def __init__(self, traces_dir: Optional[Path] = None) -> None:
        self.traces_dir = traces_dir or get_settings().traces_dir
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.main_file = self.traces_dir / "traces.jsonl"

    def save_trace(self, trace: Trace) -> Path:
        """Save a trace both to the aggregate traces.jsonl and a per-trace file."""
        trace_data = trace.model_dump(mode="json")
        append_jsonl(self.main_file, trace_data)

        # Also save individual trace for fast isolated loading
        per_trace_path = self.traces_dir / f"trace_{trace.trace_id}.json"
        with open(per_trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)

        return per_trace_path

    def load_trace(self, trace_id: str) -> Optional[Trace]:
        """Load a trace by its UUID."""
        per_trace_path = self.traces_dir / f"trace_{trace_id}.json"
        if per_trace_path.exists():
            with open(per_trace_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Trace.model_validate(data)
        return None

    def list_traces(self) -> List[Trace]:
        """List all traces in the store."""
        records = read_jsonl(self.main_file)
        return [Trace.model_validate(r) for r in records]
