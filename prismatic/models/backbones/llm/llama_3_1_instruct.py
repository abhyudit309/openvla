"""
llama_3_1_instruct.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

from typing import Optional, Sequence, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, LLaMa3InstructPromptBuilder

# Registry =>> Support LLaMa-3 Models (from HF Transformers)
# fmt: off
LLAMA3_MODELS = {
    # === LLaMa-3.1 Instruct Models ===
    "llama-3.1-8B-instruct": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Meta-Llama-3.1-8B-Instruct"
    },
}
# fmt: on


class LLaMa3InstructLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        use_lora: bool = False,
        lora_config = None,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            use_lora=use_lora,
            lora_config=lora_config,
            **LLAMA3_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-3 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("llama-3.1") and self.identifier.endswith("-instruct"):
            return LLaMa3InstructPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    # Copied from llama2.py directly. FIX later!
    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)