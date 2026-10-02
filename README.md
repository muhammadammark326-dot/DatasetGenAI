# DatasetGen AI

**DatasetGen AI** turns natural-language requests into verified, domain-validated, deduplicated datasets and logs complete **decision traces** for training specialized open-source models.

> **Guiding Principle:**
> Better Validators → Better Training Data → Better DatasetGen Model.
> *Reliably turn a natural-language request into a validated dataset before fine-tuning any model.*

---

## Architecture at a Glance

1. **Planner Layer (`datasetgen/planner/`)**: Decomposes natural language requests into an immutable Pydantic `Blueprint`.
2. **Generator Layer (`datasetgen/generator/`)**: Batched generation with quota scheduling per topic × difficulty bucket and disk-backed caching.
3. **Quality & Validation Layer (`datasetgen/validators/`)**: Programmatic verification first:
   - Schema enforcement (Pydantic types, required keys, enum boundaries)
   - Deterministic constraints (disallowed tokens, impossible physical quantities)
   - Exact deduplication (SHA-256) & near-duplicate detection (stemmed Jaccard similarity)
   - Domain verification with **SymPy** and **Pint** for physics and mathematics
4. **Targeted Regeneration**: Rejected examples trigger targeted batch requests to fill under-represented buckets until exact quotas are met.
5. **Decision Traces (`datasetgen/schemas/trace.py`)**: Stores all accepted examples, rejected examples, validator failure reasons, and regeneration attempts for downstream model fine-tuning.

---

## Local Setup (Windows PC — Offline & Lightweight)

DatasetGen AI runs completely offline with 0 API keys and no GPU requirements using `MockProvider`.

```bash
# 1. Clone repository
git clone https://github.com/<YOUR_USERNAME>/DatasetGenAI.git
cd DatasetGenAI

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install in editable mode
pip install -e . -r requirements.txt
```

---

## CLI Usage

### 1. Plan a Dataset
Inspect the generated Blueprint before committing to generation:
```bash
datasetgen plan "I need 100 high-school physics questions" --provider mock
```

### 2. Generate and Validate End-to-End
Run full batch generation, validation, targeted regeneration, and trace export:
```bash
datasetgen run "I need 100 high-school physics questions" --provider mock --yes
```

### 3. Inspect Outputs
All outputs are organized cleanly under the path defined by `DATASETGEN_DATA_DIR` (default `./data`):
- **Final Validated Dataset:** `data/datasets/physics_qa_dataset/final.jsonl`
- **Approved Blueprint:** `data/datasets/physics_qa_dataset/blueprint.json`
- **Dataset Provenance:** `data/datasets/physics_qa_dataset/provenance.json`
- **Full Decision Trace:** `data/traces/trace_<UUID>.json` & `data/traces/traces.jsonl`

---

## Running Offline Unit Tests

```bash
pytest tests/ -v
```
All 13 unit tests pass offline in under 3 seconds with zero network calls and no GPU.

---

## Google Colab & Drive Workflow

Google Colab handles GPU execution, API teacher runs, and QLoRA fine-tuning. Google Drive persists data.
See [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) for full setup instructions and open:
- `notebooks/00_colab_setup.ipynb`: Mount Drive, pull repo, install requirements.
- `notebooks/01_planner_test.ipynb`: Test request planning.
- `notebooks/04_full_pipeline.ipynb`: End-to-end dataset generation on Colab.
