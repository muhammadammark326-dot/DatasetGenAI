"""Script to populate functional Colab notebooks."""

import json
from pathlib import Path

nb_bench = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# DatasetGen AI — Benchmark Evaluation Suite\n",
                "Evaluates generator quality, domain validity, and schema adherence against fixed benchmark tasks."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from datasetgen.evaluation.benchmark_runner import BenchmarkRunner\n",
                "from datasetgen.providers import get_provider\n",
                "\n",
                "# Use mock or real API provider (e.g. gemini)\n",
                "provider = get_provider('mock')\n",
                "runner = BenchmarkRunner(provider)\n",
                "results = runner.run_all()\n",
                "\n",
                "print(f'Evaluated {len(results)} benchmark datasets.')\n",
                "for r in results:\n",
                "    status = 'PASSED' if r.passed_quality_gate else 'FAILED'\n",
                "    print(f'- {r.benchmark_id}: {r.total_accepted} accepted | Gate: {status}')\n"
            ],
            "execution_count": None,
            "outputs": []
        }
    ],
    "metadata": {"language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4,
    "nbformat_minor": 2
}

nb_sft = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# DatasetGen AI — QLoRA SFT Training on Colab GPU\n",
                "Fine-tunes a small open model (e.g. Qwen2.5 1.5B/3B) using decision traces exported from DatasetGen AI."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 1: Export traces to SFT format (train_sft.jsonl, val_sft.jsonl)\n",
                "from datasetgen.training.prepare_sft import SFTDataPreparer\n",
                "\n",
                "preparer = SFTDataPreparer()\n",
                "train_p, val_p, summary = preparer.prepare_dataset()\n",
                "print(f\"Exported {summary['total_records']} SFT training records.\")\n",
                "print(f\"Train split: {summary['train_count']}, Validation split: {summary['val_count']}\")\n"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 2: Launch QLoRA Fine-Tuning\n",
                "from datasetgen.training.train_qlora import run_qlora_training\n",
                "\n",
                "run_qlora_training(\n",
                "    model_id='Qwen/Qwen2.5-1.5B-Instruct',\n",
                "    epochs=3,\n",
                "    batch_size=4,\n",
                "    learning_rate=2e-4,\n",
                ")\n"
            ],
            "execution_count": None,
            "outputs": []
        }
    ],
    "metadata": {"language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("notebooks/05_benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_bench, f, indent=1)
with open("notebooks/06_sft_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_sft, f, indent=1)
print("Updated notebooks 05 and 06 successfully.")
