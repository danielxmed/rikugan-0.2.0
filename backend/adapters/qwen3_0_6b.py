from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.adapters import ModelAdapter, ModelInfo
from backend.hooks import ActivationAccumulator, ActivationSliceAccumulator


class Qwen3_0_6B_Adapter(ModelAdapter):
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    @staticmethod
    def info() -> ModelInfo:
        return ModelInfo(
            adapter_id="qwen3-0.6b",
            display_name="Qwen 3 0.6B",
            hf_repo="Qwen/Qwen3-0.6B",
            aliases=("qwen3-0.6b", "qwen3-0_6b", "qwen-0.6b"),
            layers=28,
            heads=16,
            kv_heads=8,
            d_model=1024,
            d_intermediate=3072,
            vocab_size=151936,
            max_seq_len=32768,
            parameters_approx="0.6B",
        )

    def load(self, device: str, dtype_str: str, cache_dir: str, hf_token: str | None) -> None:
        dtype = getattr(torch, dtype_str, torch.bfloat16)
        model_info = self.info()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_info.hf_repo,
            cache_dir=cache_dir,
            token=hf_token,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_info.hf_repo,
            torch_dtype=dtype,
            attn_implementation="eager",
            cache_dir=cache_dir,
            token=hf_token,
        )
        self._model.to(device)
        self._model.eval()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_tokens = outputs[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_with_hooks(self, prompt: str, max_new_tokens: int = 64) -> tuple[str, list[float]]:
        accumulator = ActivationAccumulator(self.info().layers)
        accumulator.install(self._model, self)
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            input_len = inputs["input_ids"].shape[1]
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_tokens = outputs[0][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            block_heat = accumulator.compute_block_heat()
        finally:
            accumulator.remove()
        return text, block_heat

    def module_path_for_layer(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}"

    def module_path_for_attn(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}.self_attn"

    def module_path_for_mlp(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}.mlp"

    @torch.inference_mode()
    def forward_with_slices(
        self, prompt: str, max_new_tokens: int = 64
    ) -> tuple[str, list[float], bytes, dict, bytes, bytes]:
        """Single forward pass for full-sequence slices, then generate for text.

        Returns (text, block_heat, slice_bytes, slice_meta, token_proj_bytes, dim_proj_bytes).
        """
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        # Forward pass with slice hooks (full sequence activations)
        accumulator = ActivationSliceAccumulator(self.info().layers)
        accumulator.install(self._model, self)
        try:
            self._model(**inputs)
            block_heat = accumulator.compute_block_heat()
            slice_bytes, slice_meta, token_proj_bytes, dim_proj_bytes = (
                accumulator.normalize_and_serialize()
            )
        finally:
            accumulator.remove()

        # Generate text (no hooks)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_tokens = outputs[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return text, block_heat, slice_bytes, slice_meta, token_proj_bytes, dim_proj_bytes
