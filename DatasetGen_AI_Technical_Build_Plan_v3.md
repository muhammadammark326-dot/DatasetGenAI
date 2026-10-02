# DatasetGen AI — Technical Build Plan (v3)

> v3 = original build plan + benchmark/evaluation additions + the concrete **Windows → GitHub → Colab → Drive** workflow and **training-data sourcing** answer.
> This file is the single source of truth. Put a copy at `docs/BUILD_PLAN.md` in the repo.

---

## 0. Guiding Principle

**Build the dataset-generation system first. Instrument it. Collect high-quality traces. Prove the metrics. Only then fine-tune a model.**

```text
Better Validators → Better Training Data → Better DatasetGen Model
      ↑                                              │
      └────────── Better Datasets / Traces ←─────────┘
```

The first goal is NOT "train my own LLM". The first goal is: **reliably turn a natural-language request into a validated dataset.**

---

## 1. Where Everything Lives

| Place | Role | Contains | Does NOT contain |
|---|---|---|---|
| **Windows PC (8 GB RAM)** + VS Code / Antigravity | Software lab | All source code, tests, API, CLI, docs. Runs with a **mock provider** or a remote API. | Model training, big local models |
| **GitHub** | Source of truth for code | Code, configs, tests, docs, notebooks (outputs cleared), schemas, tiny sample data | Datasets > ~50 MB, checkpoints, API keys |
| **Google Colab** | ML / GPU lab | Thin notebooks that `import datasetgen`; GPU generation; QLoRA SFT; benchmark runs | Business logic (it lives in the package) |
| **Google Drive** | Persistent storage | Generated datasets, traces, checkpoints, benchmark results, LoRA adapters | Source code |

```text
        WINDOWS (VS Code / Antigravity)
                    │  git push
                    ▼
                 GITHUB
                    │  git clone / pull (inside Colab)
                    ▼
              GOOGLE COLAB (GPU)
                    │  reads / writes
                    ▼
              GOOGLE DRIVE  (datasets, traces, checkpoints, results)
```

### The one rule that makes this work

**Code lives in the package. Notebooks only call it.** Never build one giant notebook.

```python
# notebook cell — thin
from datasetgen.pipeline import run_pipeline
run_pipeline(request="100 high-school physics questions", config="configs/dev.yaml")
```

### Path abstraction (critical)

All file locations come from one environment variable so identical code runs on Windows and Colab:

| Environment | `DATASETGEN_DATA_DIR` |
|---|---|
| Windows | `./data` (inside repo, git-ignored) |
| Colab | `/content/drive/MyDrive/DatasetGenAI` |

No hard-coded `C:\...` or `/content/...` paths anywhere in the code.

---

## 2. Architecture (3 Layers)

**Planner Layer** — natural-language request → structured **Blueprint**. Hybrid: regex extracts numbers/language/format; LLM extracts domain, audience, task, difficulty, constraints. Detects missing/ambiguous requirements. **User must approve the Blueprint before generation.**

**Generator Layer** — calls the LLM in **batches** (20 → 100 → 1,000), never the whole dataset at once. Quota scheduling per topic × difficulty bucket. Unified JSON schema. Supports QA, classification, instruction tuning, math, coding, translation.

**Quality Layer (core moat)** — deterministic validators first, LLM judge only where programmatic checking is impossible. Failures never enter the final dataset and trigger **targeted regeneration** for their bucket.

```text
Request → Planner → Blueprint → (user approval) → Batch Scheduler → Generator
        → Schema → Exact Dedup → Near Dedup → Constraints → Domain Validator
        → Semantic Judge → Accept / Reject → Targeted Regeneration → Final Dataset
        → Trace saved → (later) SFT dataset → Fine-tuned DatasetGen model
```

The planner and generator may be different models. Providers are swappable via an abstraction (`LLMProvider`); never hard-code one vendor.

---

## 3. Blueprint

Immutable once generation starts; any change creates a new version.

```json
{
  "dataset_name": "high_school_physics_qa",
  "dataset_version": "1.0.0",
  "blueprint_version": "1",
  "task_type": "question_answering",
  "domain": "physics",
  "education_level": "high_school",
  "language": "English",
  "number_of_examples": 10000,
  "topics": {"mechanics": 0.3, "thermodynamics": 0.2, "electricity": 0.2, "waves": 0.15, "optics": 0.15},
  "difficulty_distribution": {"beginner": 0.4, "intermediate": 0.4, "advanced": 0.2},
  "fields": [
    {"name": "question", "type": "string"},
    {"name": "answer", "type": "string"},
    {"name": "explanation", "type": "string"},
    {"name": "topic", "type": "string"},
    {"name": "difficulty", "type": "enum"}
  ],
  "output_format": "jsonl",
  "validation_rules": ["schema", "dedup", "physics_numeric"],
  "max_retries": 3,
  "random_seed": 18472,
  "generator_model": "provider/model-name",
  "prompt_version": "v1"
}
```

Also supported: style/tone constraints, source-grounding requirements, safety constraints.

---

## 4. Validators (Core Moat)

### 4.1 Hierarchy

1. **Schema** — valid JSON, required fields, types, enums, length limits
2. **Deterministic constraints** — numeric ranges, regex, required keywords, label validity, distribution
3. **Domain verification** — math/physics recomputation, sandboxed code execution, SQL execution, span checks
4. **Semantic evaluation** — LLM judge with rubric, contradiction/relevance/completeness
5. **Human review** — sample audits, borderline cases, benchmark sets

Every decision records **which validator** decided and **why**.

### 4.2 Interface

```python
validate(example, blueprint) -> ValidationResult
```

```json
{
  "status": "reject",
  "validator": "physics_energy_v1",
  "score": 0.0,
  "errors": ["computed answer does not match generated answer"],
  "metadata": {"expected": 42.5, "generated": 45.2}
}
```

### 4.3 Dataset-specific validators

| Dataset type | Method | Tools |
|---|---|---|
| Physics QA | Recompute numeric answer | SymPy, Pint |
| Math | Symbolic equivalence | SymPy |
| Code | Sandboxed execution / tests | subprocess (resource limits) → Docker later |
| Translation | Language ID + quality signals | fastText, sacreBLEU/chrF |
| Classification | Label ∈ allowed set | Python |
| Context QA | Answer supported by context | span / semantic check |
| JSON extraction | Schema + field constraints | Pydantic / JSON Schema |
| SQL | Parse + run on test DB | sqlglot + sandbox |

> **Example:** question `2x + 3 = 13`. LLM says `x = 5`. SymPy solves independently → `5`. **PASS.** The LLM is never trusted blindly.

**Security:** never run untrusted generated code on the host without isolation; enforce CPU/memory/time/network/filesystem limits.

### 4.4 Quality gates before a dataset is "READY"

Schema → Constraints → Dedup → Domain correctness → Semantic quality → Distribution check → Human sample audit → **READY**.

---

## 5. Reliability & Provenance

- **Reproducibility:** store blueprint version, model + version, prompt version, validator versions, seed, generation params, dataset version.
- **Caching:** key = `hash(blueprint + model + prompt_version + batch_index + seed)`; never pay twice.
- **Checkpointing / resume:** if a run dies at 62,000/100,000 — resume from the last good batch. *Essential on Colab because sessions disconnect.*
- **Retries:** exponential backoff, failed-batch recovery, provider fallback, per-bucket progress.
- **Observability:** job/batch status, failures, tokens, latency, cost, acceptance rate.
- **Cost metric:** *cost per accepted example*, not per generated example.
- **Provenance record per dataset:** `dataset_name, dataset_version, blueprint_version, generator_model, validator_versions, source_datasets, license, timestamp, seed`.

---

## 6. Traces (the real training asset)

The key asset is not the final dataset — it is the **complete decision trace**. **Never discard rejected examples.**

```json
{
  "trace_id": "uuid",
  "user_request": "...",
  "blueprint": {},
  "blueprint_version": "1",
  "generator_model": "...",
  "prompt_version": "v1",
  "batch_id": "...",
  "examples": [],
  "validation_results": [],
  "quality_scores": {},
  "rejected_examples": [],
  "regeneration_attempts": [],
  "human_corrections": [],
  "final_examples": [],
  "timestamp": "..."
}
```

Define this schema **before** large-scale generation. Stored as JSONL under `DATASETGEN_DATA_DIR/traces/`.

---

## 7. Benchmark & Evaluation (fixed, separate from training data)

**Initial benchmark** (~500 requests): 100 physics, 100 math, 100 coding, 100 classification, 100 context-QA. Each has known requirements and expected constraints. Store in `benchmarks/` and **never** use it to build SFT data.

**Metrics:** blueprint accuracy · schema-valid rate · constraint-satisfaction · domain-valid rate · duplicate / near-duplicate rate · topic-distribution error · difficulty-distribution error · diversity · semantic quality · human acceptance · regeneration rate · tokens per accepted example · latency · cost per 1,000 accepted examples.

**Model comparison:** base model, fine-tuned model, improved model — all run on the *same* benchmark. Do not train a new model unless you can measure whether it improved.

---

## 8. Where Does the Training Data for *Your* LLM Come From?

Your DatasetGen model is trained on **DatasetGen's own decision traces** — not on a generic internet dataset. There are three sources:

### Source A — Your own pipeline traces (primary, ~80–90 %)
A **teacher LLM** (a capable API model or a larger open model) runs *inside your pipeline*. Every run produces a trace (§6). Those traces are converted into SFT examples:

| SFT task | Input → Output |
|---|---|
| Request → Blueprint | user request → blueprint JSON |
| Blueprint → Batch | request + blueprint → structured examples |
| Repair | failed example + validator error → corrected example |
| Targeted generation | under-filled bucket → examples for that bucket |
| Constraint repair | constraint violation → corrected generation |

Only traces that **passed the quality gates** become SFT data; rejected examples become *repair* training pairs and later DPO "worse" samples.
**Target:** 5k–10k very high-quality traces first (not millions of noisy ones).

### Source B — Open datasets as *seed material* (secondary)
Used for seed topics/questions, grounding, and benchmark references — not blindly copied in.
Candidates (**verify each license and provenance before use**): GSM8K, MATH (math); SQuAD, ARC, OpenBookQA, SciQ (QA / science); MBPP, HumanEval (code); Hugging Face `datasets` hub for classification.
Some licenses are non-commercial (e.g., SciQ) or share-alike — record the license in the provenance file.

### Source C — Source-grounded synthesis (optional, later)
Textbooks, docs, or notes you are **permitted** to use → generate examples grounded in them → validate.

### Source D — Human corrections (smallest, highest value)
Your own review of borderline cases → `human_corrections` field → highest-weight training/eval data.

### Important cautions
- **Check the teacher model's terms of service** regarding using its outputs to train another model. Prefer open-weight teachers with permissive licenses if you plan to release/commercialize.
- Keep **benchmark data out of training data** (no leakage).
- Deduplicate train vs. benchmark.

```text
Teacher LLM ─► DatasetGen pipeline ─► validated traces ─► SFT dataset ─► QLoRA on Colab ─► your DatasetGen model
   ▲                                       ▲
Open datasets (seeds)               Human corrections
```

---

## 9. Training Plan (Colab only)

1. **SFT first** with LoRA/QLoRA on a **small open model** (e.g., a 1.5B–3B instruct model such as the Qwen2.5 family or similar permissive-license model) — fits a free Colab T4.
2. **Evaluate** on the fixed benchmark vs. the teacher baseline.
3. If worse: investigate data → prompts → blueprint → validators → hyper-parameters *before* making the model bigger.
4. **DPO later**, only with measurable preference pairs (validator score, constraint satisfaction, diversity, human preference).
5. **RLVR/GRPO later**, only if benchmark is stable, validators reliable, SFT baseline useful, and rewards have no obvious loopholes (beware reward hacking).
6. 7B–14B models are a long-term target, **not** the MVP.

Training artifacts (LoRA adapters, checkpoints) → saved to Drive (`checkpoints/`), **not** GitHub.
Do **not** train on the 8 GB Windows PC.

---

## 10. Repository Layout

```text
DatasetGenAI/
├── datasetgen/                  # the installable Python package (ALL logic here)
│   ├── config/                  # settings, env handling, DATASETGEN_DATA_DIR
│   ├── schemas/                 # Blueprint, Trace, ValidationResult (Pydantic)
│   ├── providers/               # base.py, mock.py, openai_compat.py, gemini.py, hf_local.py
│   ├── planner/                 # extractors.py, blueprint_builder.py
│   ├── generator/               # batcher.py, quota_scheduler.py, prompts/
│   ├── validators/              # base.py, schema.py, constraints.py, dedup.py,
│   │                            # math_validator.py, physics.py, code_sandbox.py,
│   │                            # classification.py, semantic.py, registry.py
│   ├── pipeline/                # orchestrator.py (generate→validate→regenerate loop)
│   ├── storage/                 # jsonl_io.py, trace_store.py, checkpoint.py, cache.py, versioning.py
│   ├── evaluation/              # benchmark_runner.py, metrics.py, report.py
│   ├── training/                # prepare_sft.py, train_qlora.py  (run on Colab)
│   └── cli.py                   # `datasetgen run|plan|bench|export-sft`
├── app/
│   ├── api/                     # FastAPI (later)
│   └── frontend/                # UI (later)
├── configs/                     # dev.yaml, colab.yaml, providers.yaml
├── benchmarks/                  # fixed benchmark request files (small, versioned)
├── notebooks/                   # THIN Colab notebooks (outputs cleared)
│   ├── 00_colab_setup.ipynb
│   ├── 01_planner_test.ipynb
│   ├── 02_generator_test.ipynb
│   ├── 03_validation_test.ipynb
│   ├── 04_full_pipeline.ipynb
│   ├── 05_benchmark.ipynb
│   └── 06_sft_training.ipynb
├── tests/                       # pytest (all pass offline using mock provider)
├── scripts/                     # helper scripts
├── docs/                        # BUILD_PLAN.md, ARCHITECTURE.md, COLAB_GUIDE.md
├── data/                        # LOCAL ONLY, git-ignored (datasets/, traces/, checkpoints/)
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-colab.txt
└── README.md
```

### Google Drive layout

```text
MyDrive/DatasetGenAI/
├── datasets/{physics,mathematics,coding,qa,classification}/
├── traces/
├── sft_data/
├── checkpoints/
├── experiments/
└── benchmark_results/
```

---

## 11. Colab Workflow

```python
# Cell 1 — setup (notebooks/00_colab_setup.ipynb)
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
os.environ["DATASETGEN_DATA_DIR"] = "/content/drive/MyDrive/DatasetGenAI"
os.environ["PROVIDER_API_KEY"] = userdata.get("PROVIDER_API_KEY")   # Colab Secrets, never in code

!git clone https://github.com/<YOU>/DatasetGenAI.git /content/DatasetGenAI
%cd /content/DatasetGenAI
!pip install -q -e . -r requirements-colab.txt
```

On later sessions: `git pull` instead of `clone`. Every notebook starts by running the setup cell, then calls package functions. Long jobs must checkpoint to Drive.

**Secrets:** API keys go in Colab Secrets (locally in `.env`, which is git-ignored). Never commit keys.

---

## 12. Roadmap (Build Order)

| Phase | Goal | Where |
|---|---|---|
| **0 Foundation** | Blueprint + Trace schemas, Pydantic, provider abstraction (+ mock), JSONL export, logging, caching, tests | Windows |
| **1 Text-QA MVP** | "100 high-school physics questions" → Blueprint → approval → 20-example preview → validate → regenerate → final JSONL → stats report | Windows + Colab notebooks 01–04 |
| **2 Multi-type** | Classification, math, context-QA, instruction tuning, coding — one validator module each | Windows |
| **3 Evaluation** | Fixed benchmark + automated regression runner + metrics | Windows + Colab 05 |
| **4 Trace factory** | Run teacher models → 5k–50k traces, rejected examples, corrections, preference pairs | Colab (+Drive) |
| **5 Small model SFT** | `prepare_sft.py` + `train_qlora.py`; benchmark vs. teacher | Colab 06 |
| **6 Advanced** | DPO, GRPO/RLVR, larger models, routing — only if benchmark justifies | Colab |
| **7 Scale** | Queues, parallel workers, versioning, multi-provider, production UI | Cloud |

**Example Phase-1 report:**
```text
Requested: 100   Generated: 123   Accepted: 100   Rejected: 23
Duplicates: 8    Domain failures: 10   Schema failures: 5   Semantic failures: 8
Regeneration attempts: 31
```

**Development rule:** 20 examples → validate → fix → 100 → validate → 1,000 → 10,000+.

---

## 13. Cost & Free-First Strategy

Develop with the **mock provider** (free, offline, deterministic) → test pipeline → use a free/low-cost inference tier or small open model → use a stronger teacher API **only** for high-quality trace creation → cache everything → fine-tune small model → reduce paid-API dependence.

---

## 14. Long-Term Product Vision

> "Create 50,000 grade-10 physics questions in English and Urdu, 40 % beginner, 40 % intermediate, 20 % advanced, with numerical problems in every topic."

Understand → ask for missing info → build Blueprint → user edits/approves → plan quotas → generate batches → validate → regenerate → dedup → check distributions → quality report → export JSONL/CSV/Parquet → store provenance → reproducible/versioned.

**Differentiator:** AI generation **+ programmatic verification + quality control + reproducibility + traceable provenance.**
