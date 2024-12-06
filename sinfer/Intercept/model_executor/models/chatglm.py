import torch
import sinfer
from torch import nn
from torch.nn import LayerNorm
from typing import List, Optional
from transformers import PretrainedConfig
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from sinfer.model_executor.layers.layernorm import RMSNorm
from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward
from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from sinfer.logger import init_logger

from sinfer.model_executor.models.chatglm import GLMAttention, GLMBlock

logger = init_logger(__name__)


def intercept_GLMAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.query_key_value(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    # pack q & k to fit bt_ops.apply_rotary
    qk, _ = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)
    self.rotary_emb(position_ids, qk.view(-1, self.num_heads + self.num_kv_heads, self.head_dim))

    context_layer = self.attn(
        q,
        k,
        v,
        kv_cache,
        attn_metadata,
    )
    attn_output, _ = self.dense(context_layer)
    return attn_output


def intercept_GLMBlock__init__(
    self,
    config,
    quant_config: Optional[QuantizationConfig] = None,
):
    super(GLMBlock, self).__init__()
    self.apply_residual_connection_post_layernorm = (
        config.apply_residual_connection_post_layernorm)

    self.fp32_residual_connection = config.fp32_residual_connection

    layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
    # Layernorm on the input data.
    self.input_layernorm = layer_norm_func(config.hidden_size,
                                            eps=config.layernorm_epsilon)

    # Self attention.
    self.self_attention = GLMAttention(config, quant_config)
    self.hidden_dropout = config.hidden_dropout

    # Layernorm on the attention output
    self.post_attention_layernorm = layer_norm_func(
        config.hidden_size, eps=config.layernorm_epsilon)

    # use FeedForward instead of MLP
    # MLP
    self.mlp = FeedForward(
        hidden_size=config.hidden_size,
        intermediate_size=config.ffn_hidden_size,
        hidden_act='silu',
        up_proj_name='dense_h_to_4h',
        is_gated=True,
        down_proj_name='dense_4h_to_h',
        bias=config.add_bias_linear,
        quant_config=quant_config
    )

InterceptObject.add_intercept(GLMAttention,
                                  GLMAttention.forward,
                                  intercept_GLMAttention_forward)
InterceptObject.add_intercept(GLMBlock,
                                  GLMBlock.__init__,
                                  intercept_GLMBlock__init__)