from typing import List, Optional, Tuple, Union
import torch
from src.vlm_backbone.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from src.vlm_backbone.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLDecoderLayer,
    Qwen2RMSNorm,
    Qwen2_5_VLAttention,
    Qwen2_5_VLFlashAttention2,
    Qwen2_5_VLSdpaAttention,
    Qwen2MLP,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLModel, 
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLPreTrainedModel
)
from torch import nn
from transformers.utils import logging
from peft import PeftModel

logger = logging.get_logger(__name__)


class ModifiedQwen2_5_VLAttention(Qwen2_5_VLAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2_5_VLFlashAttention2(Qwen2_5_VLFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2_5_VLSdpaAttention(Qwen2_5_VLSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": ModifiedQwen2_5_VLAttention,
    "flash_attention_2": ModifiedQwen2_5_VLFlashAttention2,
    "sdpa": ModifiedQwen2_5_VLSdpaAttention,
}


class ModifiedQwen2_5_VLDecoderLayer(Qwen2_5_VLDecoderLayer):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen2_5_VLBiModel(Qwen2_5_VLModel):
    _no_split_modules = ["ModifiedQwen2_5_VLDecoderLayer"]

    def __init__(self, config: Qwen2_5_VLConfig):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedQwen2_5_VLDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class Qwen2_5_VLBiForMNTP(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)

        self.model = Qwen2_5_VLBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None
        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)
