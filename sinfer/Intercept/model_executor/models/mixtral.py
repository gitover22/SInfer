import torch
import sinfer
from torch import nn
from bangtransformer.torch import bt_ops
from typing import List, Optional, Tuple, Union
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.distributed import tensor_model_parallel_all_reduce
from sinfer.model_executor.models.mixtral import MixtralAttention, MixtralMoE
from sinfer.Intercept.intercept_utils import set_is_gated

def intercept_MixtralAttention_forward(
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
    
    attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    output, _ = self.o_proj(attn_output)
    return output

def intercept_MixtralMoE_forward(
    self,
    hidden_states: torch.Tensor
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    hidden_states = hidden_states.view(-1, self.hidden_size)
    # router_logits: (num_tokens, n_experts)
    router_logits, _ = self.gate(hidden_states)
    # use bt_ops.fused_moe
    final_hidden_states = bt_ops.fused_moe(hidden_states,
                                           router_logits,
                                           self.w13_weight,
                                           self.w2_weight,
                                           None,
                                           None,
                                           None,
                                           None,
                                           None,
                                           None,
                                           None,
                                           self.top_k,
                                           True,
                                           True,
                                           'silu')

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(
            final_hidden_states)

    return final_hidden_states.view(num_tokens, hidden_size)


set_is_gated(True)
InterceptObject.add_intercept(MixtralAttention,
                                  MixtralAttention.forward,
                                  intercept_MixtralAttention_forward)
InterceptObject.add_intercept(MixtralMoE,
                                  MixtralMoE.forward,
                                  intercept_MixtralMoE_forward)

