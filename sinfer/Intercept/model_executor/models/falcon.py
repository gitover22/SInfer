import torch
import sinfer
from torch import nn
from typing import List, Optional, Union
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from torch.nn import LayerNorm
from transformers import FalconConfig as HF_FalconConfig
from sinfer.transformers_utils.configs import RWConfig
FalconConfig = Union[HF_FalconConfig, RWConfig]
from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.logger import init_logger

from sinfer.model_executor.models.falcon import FalconAttention, FalconDecoderLayer

logger = init_logger(__name__)


def intercept_FalconAttention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, bias = self.query_key_value(hidden_states)
    if bias is not None:
        qkv += bias
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    # pack q & k to fit bt_ops.apply_rotary
    if self.use_rotary:
        qk, _ = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)
        self.rotary_emb(positions, qk.view(-1, self.num_heads + self.num_kv_heads, self.head_dim))
    
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    attn_output, bias = self.dense(attn_output)
    return attn_output, bias


def intercept_FalconDecoderLayer__init__(
    self,
    config: FalconConfig,
    quant_config: Optional[QuantizationConfig] = None,
) -> None:
    super(FalconDecoderLayer, self).__init__()
    hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.self_attention = FalconAttention(config, quant_config)
    self.config = config

    self.reduce_row_parallel_results = not (config.new_decoder_architecture
                                            or config.parallel_attn)
    # use FeedForward instead of MLP
    self.mlp = FeedForward(hidden_size=hidden_size,
                            intermediate_size=hidden_size * 4,
                            hidden_act='gelu',
                            up_proj_name='dense_h_to_4h',
                            is_gated=False,
                            down_proj_name='dense_4h_to_h',
                            bias=config.bias,
                            quant_config=quant_config,
                            skip_bias_add=True,
                            reduce_results=self.reduce_row_parallel_results)
    
    if config.new_decoder_architecture:
        # The layer norm before self-attention
        self.ln_attn = LayerNorm(hidden_size,
                                    eps=config.layer_norm_epsilon)
        # The layer norm before the MLP
        self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
    else:
        self.input_layernorm = LayerNorm(hidden_size,
                                            eps=config.layer_norm_epsilon)
        if not config.parallel_attn:
            self.post_attention_layernorm = LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon)


InterceptObject.add_intercept(FalconAttention,
                                  FalconAttention.forward,
                                  intercept_FalconAttention_forward)
InterceptObject.add_intercept(FalconDecoderLayer,
                                  FalconDecoderLayer.__init__,
                                  intercept_FalconDecoderLayer__init__)
