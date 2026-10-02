# DatasetGen AI 🏭🧠

**DatasetGen AI** is a complete neuro-symbolic synthetic dataset generation factory. It turns natural-language requests into verified, domain-validated, deduplicated datasets and logs complete **decision traces** to fine-tune specialized open-source LLMs.

> **Guiding Principle:**
> Better Validators → Better Training Data → Better DatasetGen Model.
> *Reliably turn a natural-language request into a validated dataset before fine-tuning any model.*

---

## 🏛️ Architecture & Pipeline Flow

```text
Natural Language Request
        │
        ▼
[ 1. Planner Layer ] ──► Immutable Pydantic Blueprint (Schemas, Topics, Quotas, Validators)
        │
        ▼
[ 2. Generator Layer ] ──► Batched Generation (Teacher LLM or Fine-Tuned Model)
        │
        ▼
[ 3. Deterministic Validators ] ──► SymPy (Math), Pint (Physics), AST Sandbox (Code), Lexical (QA), Deduplication
        │
   ┌────┴───────────────────────────┐
   │ Pass                           │ Reject (Self-Repair Loop)
   ▼                                ▼
[ Final Dataset (JSONL) ]    [ Decision Traces ] ──► SFT & DPO Training Factory ──► Fine-Tuned Local LLM
```

---

## 🚀 Key Capabilities

* **Deterministic Programmatic Validators:**
  * **Physics & Units:** Dimensional analysis & numeric tolerance via Pint & SymPy.
  * **Mathematics:** Symbolic expression evaluation & equivalence proofs via SymPy.
  * **Code Sandbox:** Static AST security analysis & subprocess sandbox execution.
  * **Context-QA Grounding:** Token-level lexical grounding ratio verification.
  * **Deduplication:** Exact SHA-256 and near-duplicate stemmed Jaccard similarity.
* **Trace Decision Logging:** Full recording of rejected examples, failure errors, and repair steps.
* **Leakage-Free SFT & DPO Extraction:** Automatic extraction of multi-task training pairs with cryptographic hash protection against benchmark contamination.
* **QLoRA Fine-Tuning & Model Merging:** Automated 4-bit QLoRA training on GPU (Colab A100/T4) achieving >98% token accuracy, plus standalone weight merging.
* **Full-Stack Studio UI:** Built-in FastAPI server and dark-themed interactive web dashboard.

---

## 💻 Local Setup (Windows PC — Offline & Lightweight)

DatasetGen AI runs completely offline with 0 API keys and no GPU requirements using `MockProvider`.

```bash
# 1. Clone repository
git clone https://github.com/muhammadammark326-dot/DatasetGenAI.git
cd DatasetGenAI

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install in editable mode
pip install -e . -r requirements.txt
```

---

## 🖥️ Interactive Web Studio

Launch the full-featured web dashboard to plan blueprints, generate datasets, inspect decision traces, and view fine-tuned model telemetry:

```bash
datasetgen ui --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

## ⚙️ CLI Reference

### 1. Plan a Dataset
Inspect the generated Blueprint before committing to generation:
```bash
datasetgen plan "Generate 100 high-school physics questions" --provider mock
```

### 2. Generate and Validate End-to-End
Run full batch generation, validation, targeted regeneration, and trace export:
```bash
datasetgen run "Generate 100 high-school physics questions" --provider mock --yes
```

### 3. Run Benchmark Evaluation Suite
Evaluate models against fixed multi-domain benchmark tasks:
```bash
datasetgen bench --provider mock --limit 5
```

### 4. Export SFT & DPO Training Datasets
Convert validated pipeline traces into training data:
```bash
# Export multi-task SFT pairs (Request->Blueprint, Blueprint->Batch, Repair, Targeted)
datasetgen export-sft

# Export preference pairs (chosen vs rejected) for DPO
datasetgen export-dpo
```

### 5. Merge Fine-Tuned LoRA Adapter into Standalone Model
Merge trained adapter weights back into the base model for zero-dependency deployment (vLLM, Ollama, Hugging Face Hub):
```bash
datasetgen merge \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter data/checkpoints/datasetgen_qlora_adapter \
    --output data/merged_model \
    --device cuda
```

---

## 🧪 Automated Testing

Run the full unit test suite (offline, deterministic, under 4 seconds):

```bash
pytest tests/ -v
```
All **33 unit tests** pass with 100% offline coverage.

---

## ☁️ Google Colab & GPU Training Workflow

Google Colab handles GPU execution and QLoRA fine-tuning. Google Drive persists datasets and checkpoints.

```python
# Setup Cell in Colab
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/muhammadammark326-dot/DatasetGenAI.git /content/DatasetGenAI
%cd /content/DatasetGenAI
!pip install -q -e . -r requirements-colab.txt
```

* **Run QLoRA Training on GPU:**
  ```python
  from datasetgen.training.train_qlora import run_qlora_training
  run_qlora_training(model_id="Qwen/Qwen2.5-1.5B-Instruct", epochs=1, batch_size=2)
  ```
* **Run DPO Preference Optimization:**
  ```python
  from datasetgen.training.train_dpo import run_dpo_training
  run_dpo_training(model_id="Qwen/Qwen2.5-1.5B-Instruct", adapter_path="data/checkpoints/datasetgen_qlora_adapter")
  ```

---

## 📂 Output Artifacts

All outputs are saved cleanly under `DATASETGEN_DATA_DIR` (default `./data` or `/content/drive/MyDrive/DatasetGenAI`):
- **Validated Datasets:** `data/datasets/<name>/final.jsonl`
- **Blueprints:** `data/datasets/<name>/blueprint.json`
- **Provenance Records:** `data/datasets/<name>/provenance.json`
- **Complete Traces:** `data/traces/trace_<UUID>.json`
- **SFT Training Data:** `data/sft_data/train_sft.jsonl`
- **DPO Preference Pairs:** `data/dpo_data/train_dpo.jsonl`
- **LoRA Checkpoints:** `data/checkpoints/datasetgen_qlora_adapter`
- **Standalone Merged Models:** `data/merged_model/`
