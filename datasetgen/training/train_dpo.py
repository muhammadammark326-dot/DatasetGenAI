"""Direct Preference Optimization (DPO) fine-tuning pipeline for Colab."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from datasetgen.config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_dpo_training(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dpo_dir: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    max_length: int = 1024,
) -> Path:
    """Execute DPO training on Hugging Face transformers/TRL."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    settings = get_settings()
    dpo_path = Path(dpo_dir) if dpo_dir else (settings.data_dir / "dpo_data")
    train_file = dpo_path / "train_dpo.jsonl"
    val_file = dpo_path / "val_dpo.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(
            f"DPO training data not found at {train_file}. Run `datasetgen export-dpo` first."
        )

    out_path = Path(output_dir) if output_dir else (settings.checkpoints_dir / "datasetgen_dpo_adapter")
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting DPO training on base model: %s", model_id)
    if adapter_path:
        logger.info("Initializing from SFT adapter checkpoint: %s", adapter_path)
    logger.info("Reading DPO pairs from: %s", dpo_path)
    logger.info("Checkpoints will be saved to: %s", out_path)

    # 1. Hardware detection
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    device = "cuda" if use_cuda else "cpu"
    logger.info("Using compute device: %s (bf16=%s)", device, use_bf16)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model with 4-bit Quantization
    bnb_config = None
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if use_cuda else None,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32),
    )

    if use_cuda:
        model = prepare_model_for_kbit_training(model)

    # If starting from fine-tuned SFT adapter, load it
    if adapter_path and Path(adapter_path).exists():
        logger.info("Loading pre-trained SFT adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)

    # 4. Load DPO Datasets
    data_files = {"train": str(train_file)}
    if val_file.exists():
        data_files["validation"] = str(val_file)

    dataset = load_dataset("json", data_files=data_files)

    # 5. DPO Config
    dpo_args = DPOConfig(
        output_dir=str(out_path / "runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=(use_cuda and not use_bf16),
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Handled automatically by PEFT
        args=dpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )

    logger.info("--- Initiating DPO Training Loop ---")
    trainer.train()

    logger.info("[OK] DPO Training complete! Saving final adapter to %s", out_path)
    trainer.model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO Preference Optimization.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--dpo-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    run_dpo_training(
        model_id=args.model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        dpo_dir=args.dpo_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        beta=args.beta,
    )
