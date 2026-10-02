"""HuggingFace Local / QLoRA fine-tuned model provider."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from datasetgen.providers.base import LLMProvider, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)


class HuggingFaceLocalProvider(LLMProvider):
    """LLM provider running local Hugging Face models with optional LoRA adapter."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map

        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """Lazy loader for PyTorch and Transformers to avoid slow imports when unused."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "torch and transformers are required to use HuggingFaceLocalProvider. "
                "Install them via `pip install torch transformers accelerate`"
            ) from e

        logger.info("Loading tokenizer for %s...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }

        if torch.cuda.is_available():
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig

                    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not installed, falling back to float16 loading.")
                    model_kwargs["torch_dtype"] = torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        logger.info("Loading base model %s...", self.model_name)
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if self.adapter_path:
            logger.info("Loading fine-tuned LoRA adapter from %s...", self.adapter_path)
            try:
                from datasetgen.training.compatibility import patch_torchao_compat
                patch_torchao_compat()
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
                logger.info("Successfully merged LoRA adapter %s.", self.adapter_path)
            except ImportError as e:
                raise ImportError(
                    "peft is required to load LoRA adapters. Install via `pip install peft`"
                ) from e
        else:
            self._model = base_model

        self._model.eval()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using the loaded Hugging Face model and optional adapter."""
        self._load_model()
        import torch

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            formatted_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback formatting
            sys_prefix = f"System: {system_prompt}\n\n" if system_prompt else ""
            formatted_input = f"{sys_prefix}User: {prompt}\n\nAssistant:"

        inputs = self._tokenizer(formatted_input, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(temperature, 0.01)
            gen_kwargs["top_p"] = kwargs.get("top_p", 0.9)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][prompt_len:]
        completion_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        completion_tokens = len(generated_ids)
        total_tokens = prompt_len + completion_tokens

        return LLMResponse(
            text=completion_text,
            usage=LLMUsage(
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=0.0,  # Local inference is free
            ),
            model=f"{self.model_name}+{self.adapter_path}" if self.adapter_path else self.model_name,
            metadata={"adapter_path": self.adapter_path},
        )
