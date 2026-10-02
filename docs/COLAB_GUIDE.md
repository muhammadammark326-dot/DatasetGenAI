# Google Colab Execution Guide

DatasetGen AI strictly separates logic from execution:
- **All business logic** lives in the git repository package (`datasetgen`).
- **Google Colab** runs thin notebooks (`notebooks/00_colab_setup.ipynb` through `06_sft_training.ipynb`).
- **Google Drive** stores datasets, checkpoints, and fine-tuning adapters persistently.

## Workflow Overview

```text
  Windows PC (Software Lab)
            │
            │ git push origin main
            ▼
        GitHub Repo
            │
            │ git clone / git pull
            ▼
     Google Colab (GPU Lab)
            │
            │ Read / Write
            ▼
       Google Drive (/content/drive/MyDrive/DatasetGenAI)
```

## Step 1: Push Local Code to GitHub
On your Windows machine:
```bash
git add .
git commit -m "feat: complete Phase 0 and Phase 1"
git push origin main
```

## Step 2: Open Colab Notebook
1. Open Google Colab and upload or open `notebooks/00_colab_setup.ipynb` from your GitHub repository.
2. In Colab, add your API keys (e.g. `GEMINI_API_KEY` or `OPENAI_API_KEY`) to **Secrets** (the key icon in the left sidebar).

## Step 3: Run the Standard Setup Cell
Every notebook starts with this thin initialization:

```python
from google.colab import drive, userdata
import os

# 1. Mount persistent Google Drive
drive.mount('/content/drive')

# 2. Configure path abstraction
os.environ["DATASETGEN_DATA_DIR"] = "/content/drive/MyDrive/DatasetGenAI"

# 3. Pull code from GitHub
!git clone https://github.com/<YOUR_USERNAME>/DatasetGenAI.git /content/DatasetGenAI || (cd /content/DatasetGenAI && git pull)
%cd /content/DatasetGenAI

# 4. Install dependencies
!pip install -q -e . -r requirements-colab.txt
```

## Step 4: Run Pipelines
Now simply import and execute package pipelines:
```python
from datasetgen.pipeline import DatasetPipeline
from datasetgen.providers import get_provider

provider = get_provider("gemini", model_name="gemini-1.5-flash")
pipeline = DatasetPipeline(provider, batch_size=25)

accepted, report, trace = pipeline.run(
    "Generate 1,000 high-school physics questions with step-by-step solutions",
    approval_callback=lambda bp: True,
)
print(report.format_cli())
```
All outputs are saved directly to Google Drive under `MyDrive/DatasetGenAI/`.
