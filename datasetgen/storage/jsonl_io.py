"""JSONL reading and writing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union


def write_jsonl(path: Union[Path, str], data: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: Union[Path, str], item: Dict[str, Any]) -> None:
    """Append a single dictionary to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Union[Path, str]) -> List[Dict[str, Any]]:
    """Read all entries from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    items: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def stream_jsonl(path: Union[Path, str]) -> Iterator[Dict[str, Any]]:
    """Stream entries line by line from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
