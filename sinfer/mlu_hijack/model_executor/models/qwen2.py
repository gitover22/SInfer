import torch
import sinfer
from torch import nn
from typing import List, Optional
from transformers import Qwen2Config
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from sinfer.model_executor.layers.layernorm import RMSNorm
from sinfer.mlu_hijack.mlu_definitions import *
from sinfer.mlu_hijack.model_executor.layers.feed_forward import FeedForward
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.logger import init_logger

from sinfer.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer

logger = init_logger(__name__)


def sinfer__module_executor__models__qwen2__Qwen2Attention__forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: pack q & k to fit bt_ops.apply_rotary
    ''' 
    qk, _ = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)
    self.rotary_emb(positions, qk.view(-1, self.num_heads + self.num_kv_heads, self.head_dim))
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output, _ = self.o_proj(attn_output)
    return output


def sinfer__module_executor__models__qwen2__Qwen2DecoderLayer____init__(
    self,
    config: Qwen2Config,
    layer_idx: int,
    quant_config: Optional[QuantizationConfig] = None,
) -> None:
    super(Qwen2DecoderLayer, self).__init__()
    self.hidden_size = config.hidden_size
    # Requires transformers > 4.32.0
    rope_theta = getattr(config, "rope_theta", 1000000)
    use_sliding_window = (config.use_sliding_window
                            and layer_idx < config.max_window_layers)
    self.self_attn = Qwen2Attention(
        hidden_size=self.hidden_size,
        num_heads=config.num_attention_heads,
        max_position=config.max_position_embeddings,
        num_kv_heads=config.num_key_value_heads,
        rope_theta=rope_theta,
        use_sliding_window=use_sliding_window,
        quant_config=quant_config,
        sliding_window=config.sliding_window)
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: use FeedForward instead of MLP
    '''
    self.mlp = FeedForward(hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            hidden_act='silu',
                            up_proj_name='gate_up_proj',
                            is_gated=True,
                            down_proj_name='down_proj',
                            bias=False,
                            quant_config=quant_config)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''   
    self.input_layernorm = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)


MluHijackObject.add_hijack_object(Qwen2Attention,
                                  Qwen2Attention.forward,
                                  sinfer__module_executor__models__qwen2__Qwen2Attention__forward)
MluHijackObject.add_hijack_object(Qwen2DecoderLayer,
                                  Qwen2DecoderLayer.__init__,
                                  sinfer__module_executor__models__qwen2__Qwen2DecoderLayer____init__)