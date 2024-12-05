import torch
import sinfer
from torch import nn
from typing import List, Optional
from transformers import LlamaConfig
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from sinfer.model_executor.layers.layernorm import RMSNorm
from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from sinfer.logger import init_logger

from sinfer.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer

logger = init_logger(__name__)


def sinfer__module_executor__models__llama__LlamaAttention__forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    
    # pack q & k to fit bt_ops.apply_rotary
    qk, _ = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)
    self.rotary_emb(positions, qk.view(-1, self.num_heads + self.num_kv_heads, self.head_dim))
    
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                            self.kv_scale)
    output, _ = self.o_proj(attn_output)
    return output


def sinfer__module_executor__models__llama__LlamaDecoderLayer____init__(
    self,
    config: LlamaConfig,
    quant_config: Optional[QuantizationConfig] = None,
) -> None:
    super(LlamaDecoderLayer, self).__init__()
    self.hidden_size = config.hidden_size
    rope_theta = getattr(config, "rope_theta", 10000)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None):
        rope_scaling["original_max_position_embeddings"] = (
            config.original_max_position_embeddings)
    max_position_embeddings = getattr(config, "max_position_embeddings",
                                        8192)
    sliding_window = getattr(config, "sliding_window", None)
    # Support abacusai/Smaug-72B-v0.1 with attention_bias
    # Support internlm/internlm-7b with bias
    attention_bias = getattr(config, "attention_bias", False) or getattr(
        config, "bias", False)
    self.self_attn = LlamaAttention(
        hidden_size=self.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=getattr(config, "num_key_value_heads",
                                config.num_attention_heads),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        max_position_embeddings=max_position_embeddings,
        quant_config=quant_config,
        bias=attention_bias,
        sliding_window=sliding_window,
    )
    
    # use FeedForward instead of MLP
    self.mlp = FeedForward(hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            hidden_act='silu',
                            up_proj_name='gate_up_proj',
                            is_gated=True,
                            down_proj_name='down_proj',
                            bias=False,
                            quant_config=quant_config)

    self.input_layernorm = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)


MluHijackObject.add_hijack_object(LlamaAttention,
                                  LlamaAttention.forward,
                                  sinfer__module_executor__models__llama__LlamaAttention__forward)
MluHijackObject.add_hijack_object(LlamaDecoderLayer,
                                  LlamaDecoderLayer.__init__,
                                  sinfer__module_executor__models__llama__LlamaDecoderLayer____init__)