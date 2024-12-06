import torch
import sinfer
from torch import nn
from typing import List, Optional
from transformers import PretrainedConfig
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from sinfer.model_executor.layers.layernorm import RMSNorm
from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.logger import init_logger

from sinfer.model_executor.models.qwen import QWenAttention, QWenBlock

logger = init_logger(__name__)


def intercept_QwenAttention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.c_attn(hidden_states)
    q, k, v = qkv.chunk(chunks=3, dim=-1)
    
    # pack q & k to fit bt_ops.apply_rotary
    qk, _ = qkv.split([self.head_dim * self.num_heads * 2, self.head_dim * self.num_heads], dim=-1)
    self.rotary_emb(positions, qk.view(-1, self.num_heads + self.num_heads, self.head_dim))
  
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output, _ = self.c_proj(attn_output)
    return output


def intercept_QWenBlock__init__(
    self,
    config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig] = None,
):
    super(QWenBlock, self).__init__()
    self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    rope_theta = getattr(config, "rope_theta", 10000)
    rope_scaling = getattr(config, "rope_scaling", None)
    self.attn = QWenAttention(config.hidden_size,
                              config.num_attention_heads,
                              config.max_position_embeddings,
                              rope_theta=rope_theta,
                              rope_scaling=rope_scaling,
                              quant_config=quant_config)

    self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    # use FeedForward instead of MLP
    self.mlp = FeedForward(hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size // 2,
                            hidden_act='silu',
                            up_proj_name='gate_up_proj',
                            is_gated=True,
                            down_proj_name='c_proj',
                            bias=False,
                            quant_config=quant_config)

InterceptObject.add_intercept(QWenAttention,
                                  QWenAttention.forward,
                                  intercept_QwenAttention_forward)
InterceptObject.add_intercept(QWenBlock,
                                  QWenBlock.__init__,
                                  intercept_QWenBlock__init__)