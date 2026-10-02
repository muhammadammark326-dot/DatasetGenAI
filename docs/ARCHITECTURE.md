# DatasetGen AI — System Architecture

DatasetGen AI is an offline-first, reproducible synthetic dataset generation factory with deterministic programmatic validation and complete decision telemetry.

## System Topology

```text
[ Natural Language Request ]
            │
            ▼
┌─────────────────────────┐
│     PLANNER LAYER       │──► Extracts counts, language, format & domain
│ (extractors.py, LLM)    │──► Generates immutable Pydantic Blueprint
└─────────────────────────┘
            │
            ▼
[ User Approval Gate ] (CLI prompt / --yes)
            │
            ▼
┌─────────────────────────┐
│     GENERATOR LAYER     │──► Quota Scheduler balances topic × difficulty buckets
│ (batcher.py, cache.py)  │──► Queries LLMProvider in discrete batches (20-100)
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│      QUALITY LAYER      │──► 1. Schema Validator (Pydantic types, required keys)
│    (Core Moat)          │──► 2. Deterministic Constraints (non-negative physics, style)
│                         │──► 3. Dedup (Exact SHA256 & Stemmed Near-Duplicate Jaccard)
│                         │──► 4. Domain Verification (SymPy physics & math equivalence)
│                         │──► 5. Semantic Judge (Rubric scoring for qualitative tasks)
└─────────────────────────┘
       │           │
     Pass        Reject
       │           │
       ▼           ▼
[ Accepted ]  [ Targeted Regeneration Loop ]
       │           │
       ▼           ▼
┌────────────────────────────────────────────────────────┐
│                   STORAGE & PROVENANCE                 │
│ • Datasets: final.jsonl, blueprint.json, provenance    │
│ • Decision Traces: traces/traces.jsonl & trace_*.json  │
│ • Resilient Checkpoints: checkpoints/checkpoint_*.json │
└────────────────────────────────────────────────────────┘
```

## Module Hierarchy

- `datasetgen/config/settings.py`: Central path abstraction via `DATASETGEN_DATA_DIR` (no hard-coded paths).
- `datasetgen/schemas/`: Pydantic v2 data models for `Blueprint`, `GeneratedExample`, `ValidationResult`, and `Trace`.
- `datasetgen/providers/`: Swappable LLM providers (`MockProvider`, `OpenAICompatProvider`, `GeminiProvider`).
- `datasetgen/validators/`: Programmatic validation chain (`schema`, `constraints`, `dedup`, `physics_numeric`, `math_symbolic`).
- `datasetgen/pipeline/`: `DatasetPipeline` orchestrator driving generation, validation, targeted regeneration, telemetry, and export.
- `datasetgen/storage/`: Disk-backed caching, checkpointing, and trace persistence.
