import torch
import sinfer
from torch import nn
import torch.nn.functional as F
from bangtransformer.torch import bt_ops
from typing import List, Optional, Tuple, Union, Iterable
from transformers import PretrainedConfig
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from sinfer.model_executor.layers.layernorm import RMSNorm
from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from sinfer.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from sinfer.model_executor.utils import set_weight_attrs
from sinfer.model_executor.model_loader.weight_utils import default_weight_loader
from sinfer.model_executor.layers.linear import ReplicatedLinear
from sinfer.model_executor.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock, Qwen2MoeAttention, Qwen2MoeDecoderLayer, Qwen2MoeModel, Qwen2MoeForCausalLM

def sinfer__module_executor__models__qwen2moe__Qwen2MoeAttention__forward(
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

def sinfer__module_executor__models__qwen2moe__Qwen2MoeDecoderLayer____init__(
    self,
    config: PretrainedConfig,
    layer_idx: int,
    quant_config: Optional[QuantizationConfig] = None,
) -> None:
    super(Qwen2MoeDecoderLayer, self).__init__()
    self.hidden_size = config.hidden_size
    rope_theta = getattr(config, "rope_theta", 10000)
    rope_scaling = getattr(config, "rope_scaling", None)
    max_position_embeddings = getattr(config, "max_position_embeddings",
                                      8192)
    self.self_attn = Qwen2MoeAttention(
        hidden_size=self.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        max_position_embeddings=max_position_embeddings,
        quant_config=quant_config,
    )
    if (config.num_experts is not None
            and (layer_idx + 1) % config.decoder_sparse_step == 0):
        self.mlp = Qwen2MoeSparseMoeBlock(config=config,
                                          quant_config=quant_config)
    else:

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

def sinfer__module_executor__models__qwen2moe__Qwen2MoeSparseMoeBlock____init__(
    self,
    config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig] = None,
):
    super(Qwen2MoeSparseMoeBlock, self).__init__()
    self.config = config
    self.rank = get_tensor_model_parallel_rank()
    self.tp_size = get_tensor_model_parallel_world_size()
    self.n_routed_experts = config.num_experts
    self.top_k = config.num_experts_per_tok
    if self.tp_size > self.n_routed_experts:
        raise ValueError(
            f"Tensor parallel size {self.tp_size} is greater than "
            f"the number of experts {self.n_routed_experts}.")

    # replace weight_loader for experts
    self.moe_intermediate_size = self.config.moe_intermediate_size // self.tp_size
    self.w1 = nn.Parameter(
        torch.empty(self.config.num_experts,
                    2 * self.moe_intermediate_size,
                    self.config.hidden_size,
                    dtype=torch.get_default_dtype()))
    self.w2 = nn.Parameter(
        torch.empty(self.config.num_experts,
                    self.config.hidden_size,
                    self.moe_intermediate_size,
                    dtype=torch.get_default_dtype()))

    set_weight_attrs(self.w1, {
        "weight_loader": self.weight_loader,
    })
    set_weight_attrs(self.w2, {
        "weight_loader": self.weight_loader,
    })

    self.gate = ReplicatedLinear(config.hidden_size,
                                 self.n_routed_experts,
                                 bias=False,
                                 quant_config=None)
    if config.shared_expert_intermediate_size > 0:
        # use FeedForward instead of MLP
        self.shared_expert = FeedForward(hidden_size=config.hidden_size,
                                         intermediate_size=config.shared_expert_intermediate_size,
                                         hidden_act='silu',
                                         up_proj_name='gate_up_proj',
                                         is_gated=True,
                                         down_proj_name='down_proj',
                                         bias=False,
                                         quant_config=quant_config,
                                         reduce_results=False)
    
    else:
        self.shared_expert = None
    self.shared_expert_gate = torch.nn.Linear(config.hidden_size,
                                              1,
                                              bias=False)

def sinfer__module_executor__models__qwen2moe__Qwen2MoeSparseMoeBlock__forward(
    self,
    hidden_states: torch.Tensor
) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    shared_output = None
    if self.shared_expert is not None:
        shared_output = self.shared_expert(hidden_states)
        if self.shared_expert_gate is not None:
            shared_output = F.sigmoid(
                self.shared_expert_gate(hidden_states)) * shared_output

    # router_logits: (num_tokens, n_experts)
    router_logits, _ = self.gate(hidden_states)
    # use bt_ops.fused_moe 
    final_hidden_states = bt_ops.fused_moe(hidden_states,
                                           router_logits,
                                           self.w1,
                                           self.w2,
                                           None,
                                           None,
                                           None,
                                           None,
                                           None,
                                           None,
                                           None,
                                           self.top_k,
                                           self.config.norm_topk_prob,
                                           True,
                                           'silu')

    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output
    final_hidden_states = tensor_model_parallel_all_reduce(
        final_hidden_states)

    return final_hidden_states.view(num_tokens, hidden_dim)

def sinfer__module_executor__models__qwen2moe__Qwen2MoeSparseMoeBlock__weight_loader(
    self,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    expert_id: int
):
    tp_rank = get_tensor_model_parallel_rank()
    param_data = param.data
    shard_size = self.moe_intermediate_size
    shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
    if weight_name.endswith("gate_proj.weight"):
        param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
    if weight_name.endswith("up_proj.weight"):
        param_data[expert_id,
                   shard_size:2 * shard_size, :] = loaded_weight[shard, :]
    if weight_name.endswith("down_proj.weight"):
        param_data[expert_id, :, :] = loaded_weight[:, shard]

def sinfer__module_executor__models__qwen2moe__Qwen2MoeForCausalLM__load_weights(
    self,
    weights: Iterable[Tuple[str, torch.Tensor]]
):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # load weights for experts
    expert_params_mapping = [
        # These are the weights for the experts
        # (param_name, weight_name, expert_id)
        ("w1" if weight_name in ["gate_proj", "up_proj"] else "w2",
         f"experts.{expert_id}.{weight_name}.weight", expert_id)
        for expert_id in range(self.config.num_experts)
        for weight_name in ["gate_proj", "up_proj", "down_proj"]
    ]

    params_dict = dict(self.named_parameters())
    for name, loaded_weight in weights:
        if "mlp.experts." in name:
            # load weights for experts
            for param_name, weight_name, expert_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                # name not in params_dict: Skip experts that are not assigned to this worker.
                break
        
        else:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if (("mlp.experts." in name or "mlp.shared_expert." in name)
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if (("mlp.experts." in name or "mlp.shared_expert." in name)
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

MluHijackObject.add_hijack_object(Qwen2MoeAttention,
                                  Qwen2MoeAttention.forward,
                                  sinfer__module_executor__models__qwen2moe__Qwen2MoeAttention__forward)
MluHijackObject.add_hijack_object(Qwen2MoeDecoderLayer,
                                  Qwen2MoeDecoderLayer.__init__,
                                  sinfer__module_executor__models__qwen2moe__Qwen2MoeDecoderLayer____init__)
MluHijackObject.add_hijack_object(Qwen2MoeSparseMoeBlock,
                                  Qwen2MoeSparseMoeBlock.__init__,
                                  sinfer__module_executor__models__qwen2moe__Qwen2MoeSparseMoeBlock____init__)
MluHijackObject.add_hijack_object(Qwen2MoeSparseMoeBlock,
                                  Qwen2MoeSparseMoeBlock.forward,
                                  sinfer__module_executor__models__qwen2moe__Qwen2MoeSparseMoeBlock__forward)
MluHijackObject.add_hijack_object(Qwen2MoeSparseMoeBlock,
                                  "weight_loader",
                                  sinfer__module_executor__models__qwen2moe__Qwen2MoeSparseMoeBlock__weight_loader)
MluHijackObject.add_hijack_object(Qwen2MoeForCausalLM,
                                  Qwen2MoeForCausalLM.load_weights,
                                  sinfer__module_executor__models__qwen2moe__Qwen2MoeForCausalLM__load_weights)

