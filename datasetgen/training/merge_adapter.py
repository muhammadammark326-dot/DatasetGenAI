"""Utilities to merge LoRA adapters back into base models for zero-dependency deployment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_lora_to_standalone(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
    device: str = "cpu",
) -> Path:
    """Merge LoRA weights into the base model and save as a standalone HF checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer for %s...", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    torch_dtype = torch.float16 if (device == "cuda" or torch.cuda.is_available()) else torch.float32

    logger.info("Loading base model %s (dtype=%s)...", base_model_id, torch_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    logger.info("Applying LoRA adapter from %s...", adapter_path)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging weights into standalone model...")
    merged_model = peft_model.merge_and_unload()

    logger.info("Saving merged model to %s...", out_dir)
    merged_model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    logger.info("[OK] Successfully exported standalone merged model to %s", out_dir)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    merge_lora_to_standalone(
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        output_path=args.output,
        device=args.device,
    )
