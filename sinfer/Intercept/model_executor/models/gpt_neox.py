
import torch
import sinfer
from torch import nn
from typing import List, Optional
from transformers import GPTNeoXConfig
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from sinfer.model_executor.layers.layernorm import RMSNorm
from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward
from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from sinfer.logger import init_logger

from sinfer.model_executor.models.gpt_neox import GPTNeoXAttention, GPTNeoXLayer

logger = init_logger(__name__)


def intercept_GPTNeoXAttention_forward(
    self,
    position_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.query_key_value(hidden_states)
    q, k, v = qkv.chunk(chunks=3, dim=-1)
    # pack q & k to fit bt_ops.apply_rotary
    qk, _ = qkv.split([self.num_heads * self.head_size * 2, self.num_heads * self.head_size], dim=-1)
    self.rotary_emb(position_ids, qk.view(-1, self.num_heads + self.num_heads, self.head_size))
    
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output, _ = self.dense(attn_output)
    return output


def intercept_GPTNeoXLayer__init__(
    self,
    config: GPTNeoXConfig,
    quant_config: Optional[QuantizationConfig] = None,
):
    super(GPTNeoXLayer, self).__init__()
    self.use_parallel_residual = config.use_parallel_residual
    self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
    self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                    eps=config.layer_norm_eps)
    self.attention = GPTNeoXAttention(config, quant_config)
    # use FeedForward instead of MLP
    self.mlp = FeedForward(hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            hidden_act='gelu',
                            up_proj_name='dense_h_to_4h',
                            is_gated=False,
                            down_proj_name='dense_4h_to_h',
                            bias=True,
                            quant_config=quant_config)


InterceptObject.add_intercept(GPTNeoXAttention,
                                  GPTNeoXAttention.forward,
                                  intercept_GPTNeoXAttention_forward)
InterceptObject.add_intercept(GPTNeoXLayer,
                                  GPTNeoXLayer.__init__,
                                  intercept_GPTNeoXLayer__init__)