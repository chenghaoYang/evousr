"""Text-only Qwen3.5 loading and cache metadata for vLLM 0.18."""

from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from .checkpoint import text_weights


class Qwen35Text(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    def get_mrope_input_positions(self, input_tokens, mm_features):
        import torch

        if mm_features or not input_tokens:
            raise ValueError("Expected a nonempty text-only prompt")
        return torch.arange(len(input_tokens), dtype=torch.int64).expand(3, -1).clone(), 0

    def load_weights(self, weights):
        return super().load_weights(text_weights(weights))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
