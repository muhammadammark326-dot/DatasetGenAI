"""QLoRA fine-tuning runner for Google Colab GPU (Phase 5).

Trains a lightweight open instruct model (e.g., Qwen2.5 1.5B/3B) on DatasetGen traces.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasetgen.config.settings import get_settings


def run_qlora_training(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
) -> None:
    """Execute QLoRA fine-tuning using Hugging Face PEFT + TRL on Colab GPU."""
    settings = get_settings()
    data_dir = settings.sft_data_dir
    train_file = data_dir / "train_sft.jsonl"
    val_file = data_dir / "val_sft.jsonl"

    # Auto-detect SFT dataset location across possible Colab working directories
    if not train_file.exists():
        candidate_dirs = [
            Path("/content/data/sft_data"),
            Path("/content/drive/MyDrive/DatasetGenAI/sft_data"),
            Path("data/sft_data"),
            Path("../data/sft_data"),
            Path("/content/DatasetGenAI/data/sft_data"),
        ]
        for d in candidate_dirs:
            if (d / "train_sft.jsonl").exists():
                data_dir = d
                train_file = d / "train_sft.jsonl"
                val_file = d / "val_sft.jsonl"
                break

    if not train_file.exists():
        raise FileNotFoundError(
            f"SFT training data not found at {train_file}. Run `datasetgen export-sft` first."
        )

    output_dir = settings.checkpoints_dir / "datasetgen_qlora_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting QLoRA training on model: {model_id}")
    print(f"Reading SFT data from: {data_dir}")
    print(f"Checkpoints will be saved to: {output_dir}")

    from datasetgen.training.compatibility import patch_torchao_compat
    patch_torchao_compat()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        print("\n[!] Installing missing ML dependencies...")
        os.system("pip install -q trl peft bitsandbytes accelerate datasets")
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = (device == "cuda" and not use_bf16)
    print(f"Using compute device: {device} (bf16={use_bf16}, fp16={use_fp16})")

    # 1. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if device == "cuda" else None

    # 2. Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"quantization_config": bnb_config} if bnb_config else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32),
        **model_kwargs,
    )

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # 3. LoRA Adapter Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Load & Pre-format Dataset into explicit string text column
    raw_dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})

    eos_token = tokenizer.eos_token or "<|im_end|>"

    def format_row(row: dict) -> dict:
        inst = row.get("instruction", "")
        resp = row.get("response", "")
        text = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{resp}{eos_token}"
        return {"text": text}

    dataset = raw_dataset.map(format_row, desc="Formatting SFT text rows")

    # 5. Training Arguments
    config_params = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 2,
        "learning_rate": learning_rate,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": "none",
        "dataset_text_field": "text",
    }

    try:
        sft_config = SFTConfig(max_length=max_seq_length, **config_params)
    except TypeError:
        sft_config = SFTConfig(max_seq_length=max_seq_length, **config_params)

    # Instantiate Trainer without formatting_func since dataset already has text strings
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            processing_class=tokenizer,
            args=sft_config,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=sft_config,
        )

    print("\n--- Initiating Training Loop on GPU ---")
    trainer.train()

    print(f"\n[OK] Training complete! Saving LoRA adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Your fine-tuned DatasetGen model is saved and ready!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetGen AI QLoRA Training Runner")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model Hugging Face ID")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per device batch size")
    args = parser.parse_args()

    run_qlora_training(
        model_id=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
