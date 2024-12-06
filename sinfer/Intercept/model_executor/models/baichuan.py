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

from sinfer.model_executor.models.baichuan import BaiChuanAttention, BaiChuanDecoderLayer

logger = init_logger(__name__)


def intercept_BaiChuanAttention__forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.W_pack(hidden_states)
    q, k, v = qkv.chunk(chunks=3, dim=-1)
    # pack q & k to fit bt_ops.apply_rotary
    qk, _ = qkv.split([self.num_heads * self.head_dim * 2, self.num_heads * self.head_dim], dim=-1)
    if self.postion_embedding != "ALIBI":
        self.rotary_emb(positions, qk.view(-1, self.num_heads + self.num_heads, self.head_dim))
     
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output, _ = self.o_proj(attn_output)
    return output


def intercept_BaiChuanDecoderLayer__init__(
    self,
    config: PretrainedConfig,
    position_embedding: str,
    quant_config: Optional[QuantizationConfig] = None
):
    super(BaiChuanDecoderLayer, self).__init__()
    self.hidden_size = config.hidden_size
    rope_theta = getattr(config, "rope_theta", 10000)
    max_position_embeddings = getattr(config, "max_position_embeddings",
                                        8192)
    self.self_attn = BaiChuanAttention(
        hidden_size=self.hidden_size,
        num_heads=config.num_attention_heads,
        position_embedding=position_embedding,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        quant_config=quant_config,
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


InterceptObject.add_intercept(BaiChuanAttention,
                                  BaiChuanAttention.forward,
                                  intercept_BaiChuanAttention__forward)
InterceptObject.add_intercept(BaiChuanDecoderLayer,
                                  BaiChuanDecoderLayer.__init__,
                                  intercept_BaiChuanDecoderLayer__init__)