from typing import Any, Dict, Union, List, Optional, Tuple
import math
import torch
from torch import nn
from bangtransformer.torch import bt_ops
from sinfer.Intercept.transformers_utils.configs import CustomConfig
from sinfer.attention import Attention, AttentionMetadata
from sinfer.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from sinfer.model_executor.layers.layernorm import RMSNorm
from sinfer.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear,
                                               ReplicatedLinear)
from sinfer.model_executor.layers.logits_processor import LogitsProcessor
from sinfer.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from sinfer.model_executor.layers.rotary_embedding import get_rope
from sinfer.model_executor.layers.sampler import Sampler
from sinfer.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from sinfer.model_executor.sampling_metadata import SamplingMetadata
from sinfer.request import SamplerOutput
from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2**math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2**(-(2**-(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2,
                                  total_num_heads - closest_power_of_2)
        extra_powers = torch.arange(start=1,
                                    end=1 + 2 * num_remaining_heads,
                                    step=2,
                                    dtype=torch.int32)
        slopes = torch.cat(
            [slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


class LayerNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.view(-1, self.weight.data.shape[0])
        if residual is not None:
            residual = residual.view(-1, self.weight.data.shape[0])
            return bt_ops.fused_layer_norm(x, residual, self.weight.data, self.bias.data, None, self.variance_epsilon, True)
        else:
            return bt_ops.fused_layer_norm(x, residual, self.weight.data, self.bias.data, None, self.variance_epsilon, False)


_NORM_DICT: Dict[str, nn.Module] = {"rmsnorm": RMSNorm, "layernorm": LayerNorm}


class CustomMoeBlock(nn.Module):

    def __init__(
        self,
        config: CustomConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}.")

        self.moe_intermediate_size = self.config.moe_intermediate_size // self.tp_size
        
        if quant_config is None:
            self.w1 = nn.Parameter(
                torch.empty(self.config.num_experts,
                            2 * self.moe_intermediate_size if self.config.is_gated else self.moe_intermediate_size,
                            self.config.hidden_size,
                            dtype=torch.get_default_dtype()), requires_grad=False)
            self.w2 = nn.Parameter(
                torch.empty(self.config.num_experts,
                            self.config.hidden_size,
                            self.moe_intermediate_size,
                            dtype=torch.get_default_dtype()), requires_grad=False)
            self.w1_scale = None
            self.w2_scale = None
            self.input_smooth = None
            self.act_smooth = None
        else:
            assert quant_config.weight_bits == 8
            self.w1 = nn.Parameter(
                torch.empty(self.config.num_experts,
                            2 * self.moe_intermediate_size if self.config.is_gated else self.moe_intermediate_size,
                            self.config.hidden_size,
                            device="cuda",
                            dtype=torch.int8), requires_grad=False)
            self.w2 = nn.Parameter(
                torch.empty(self.config.num_experts,
                            self.config.hidden_size,
                            self.moe_intermediate_size,
                            device="cuda",
                            dtype=torch.int8), requires_grad=False)
            self.w1_scale = nn.Parameter(
                torch.empty(
                    self.config.num_experts,
                    2 * self.moe_intermediate_size if self.config.is_gated else self.moe_intermediate_size,
                    device="cuda",
                    dtype=torch.get_default_dtype()), requires_grad=False)
            self.w2_scale = nn.Parameter(
                torch.empty(
                    self.config.num_experts,
                    self.config.hidden_size,
                    device="cuda",
                    dtype=torch.get_default_dtype()), requires_grad=False)
            self.input_smooth = None
            self.act_smooth = None
            if quant_config.quant_mode == "SmoothQuant":
                self.input_smooth =nn.Parameter(
                    torch.empty(
                        self.config.hidden_size,
                        device="cuda",
                        dtype=torch.float32), requires_grad=False)
                self.act_smooth =nn.Parameter(
                    torch.empty(
                        self.moe_intermediate_size,
                        device="cuda",
                        dtype=torch.float32), requires_grad=False)

        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.n_routed_experts,
                                     bias=False,
                                     quant_config=None)
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = FeedForward(hidden_size=config.hidden_size,
                                             intermediate_size=config.shared_expert_intermediate_size,
                                             hidden_act = self.config.hidden_act,
                                             up_proj_name='gate_up_proj',
                                             is_gated=self.config.is_gated,
                                             down_proj_name='down_proj',
                                             bias=False,
                                             quant_config=quant_config,
                                             reduce_results=False)
        else:
            self.shared_expert = None
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size,
                                                  1,
                                                  bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        final_hidden_states = bt_ops.fused_moe(hidden_states,
                                               router_logits,
                                               self.w1,
                                               self.w2,
                                               None,
                                               None,
                                               None,
                                               self.input_smooth,
                                               self.act_smooth,
                                               self.w1_scale,
                                               self.w2_scale,
                                               self.top_k,
                                               self.config.norm_topk_prob,
                                               self.config.is_gated,
                                               self.config.hidden_act)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
            
        reduce_results = (self.config.use_parallel_residual == False)
        if reduce_results:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class CustomAttention(nn.Module):

    def __init__(
        self,
        config: CustomConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        sliding_window = getattr(config, "sliding_window", None)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.kv_scale = 1.0
        self.scaling = self.head_dim**-0.5
        
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            reduce_results = (self.config.use_parallel_residual == False),
        )
        
        self.alibi_slopes = None
        self.rotary_emb = None
        if self.config.position_embedding_type == "ALIBI":
            tp_rank = get_tensor_model_parallel_rank()
            head_start = tp_rank * self.num_heads
            head_end = (tp_rank + 1) * self.num_heads
            alibi_slopes = _get_alibi_slopes(self.total_num_heads)
            self.alibi_slopes = alibi_slopes[head_start:head_end].tolist()
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
            if rope_scaling is not None and getattr(
                    config, "original_max_position_embeddings", None):
                rope_scaling["original_max_position_embeddings"] = (
                    config.original_max_position_embeddings)
            max_position_embeddings = getattr(config, "max_sequence_length", 8192)
            is_neox_style = getattr(config, "is_neox_style", False)
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                is_neox_style=is_neox_style,
                rope_scaling=rope_scaling,
            )
         
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              alibi_slopes=self.alibi_slopes,
                              sliding_window=sliding_window)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.rotary_emb:
            qk, _ = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)
            self.rotary_emb(positions, qk.view(-1, self.num_heads + self.num_kv_heads, self.head_dim))
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                self.kv_scale)
        output, _ = self.o_proj(attn_output)
        return output


class CustomDecoderLayer(nn.Module):

    def __init__(
        self,
        config: CustomConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.self_attn = CustomAttention(
            config=config,
            quant_config=quant_config,
        )
 
        mlp_bias = getattr(config, "mlp_bias", False) or getattr(config, "bias", False)
        is_gated = getattr(config, "is_gated", False)
        
        if config.num_experts is not None:
            self.mlp = CustomMoeBlock(config=config,
                                    quant_config=quant_config)
        else:
            self.mlp = FeedForward(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=self.config.hidden_act,
                                   up_proj_name='up_proj',
                                   is_gated=is_gated,
                                   down_proj_name='down_proj',
                                   bias=mlp_bias,
                                   quant_config=quant_config,
                                   reduce_results = (self.config.use_parallel_residual == False))
        
        self.input_layernorm = _NORM_DICT[self.config.norm_type](config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = _NORM_DICT[self.config.norm_type](config.hidden_size, eps=config.norm_eps)
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.use_parallel_residual:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            layernorm_output = self.input_layernorm(hidden_states)
            attention_output = self.self_attn(
                positions=positions,
                hidden_states=layernorm_output,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )
            layernorm_output = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(layernorm_output)
            
            if get_tensor_model_parallel_rank == 0:
                hidden_states = mlp_output + attention_output + hidden_states
            else:
                hidden_states = mlp_output + attention_output
            
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            return hidden_states, None
        else:
            # rmsnorm use fused_rms_norm to get better performance
            # if apply_residual_connection_post_layernorm:
            #     x = ln1(x) + attn(ln1(x))
            #     x = ln2(x) + mlp(ln2(x))
            # else:
            #     x = x + attn(ln1(x))
            #     x = x + mlp(ln2(x))
            # Self Attention
            if residual is None:
                layernorm_input = hidden_states
                layernorm_output = self.input_layernorm(layernorm_input)
            else:
                layernorm_output, layernorm_input = self.input_layernorm(
                    hidden_states, residual)
            attention_output = self.self_attn(
                positions=positions,
                hidden_states=layernorm_output,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )
            
            # Residual connection.
            if self.config.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input
            
            # Fully Connected
            layernorm_output, layernorm_input = self.post_attention_layernorm(
                attention_output, residual)
            output = self.mlp(layernorm_output)
            
            # Second residual connection.
            if self.config.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input
            
            return output, residual


class CustomModel(nn.Module):

    def __init__(
        self,
        config: CustomConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        embed_layer = VocabParallelEmbedding if self.config.use_parallel_embedding else nn.Embedding
        self.embed_tokens = embed_layer(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            CustomDecoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = _NORM_DICT[self.config.norm_type](config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class CustomForCausalLM(nn.Module):
    
    def __init__(
        self,
        config: CustomConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._verify_params()
        self.model = CustomModel(config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def _verify_params(self) -> None:
        if (self.config.max_sequence_length) is None or \
           (self.config.num_hidden_layers) is None or \
           (self.config.hidden_size) is None or \
           (self.config.vocab_size) is None or \
           (self.config.num_attention_heads) is None:
            raise ValueError(
                "max_sequence_length, num_hidden_layers, hidden_size, vocab_size, "
                "num_attention_heads, must be vaild int values")
        
        if self.config.hidden_act not in ["silu", "gelu"]:
            raise ValueError(
                "CustomConfig hidden_act must be one of [silu, gelu]. Got "
                f"{self.config.hidden_act}.")
        
        if self.config.position_embedding_type not in ["ALIBI", "ROPE"]:
            raise ValueError(
                "position_embedding_type must be one of [ALIBI, ROPE]. Got "
                f"{self.config.position_embedding_type}.")
            
        if self.config.num_experts is not None:
            if self.config.num_experts_per_tok is None:
                raise ValueError(
                    "num_experts_per_tok must be a valid int value when num_experts is not None")
            if self.config.moe_intermediate_size is None:
                raise ValueError(
                    "moe_intermediate_size must be a valid int value when num_experts is not None")
            if self.config.shared_expert_intermediate_size is None:
                raise ValueError(
                    "shared_expert_intermediate_size must be a valid int value when num_experts is not None")
            if self.config.norm_topk_prob is None:
                raise ValueError(
                    "norm_topk_prob must be a valid bool value when num_experts is not None")
            if self.config.mlp_bias is True:
                raise ValueError(
                    "mlp_bias must be False when num_experts is not None")
        else:
            if self.config.intermediate_size is None:
                raise ValueError(
                    "intermediate_size must be a valid int value when num_experts is None")

        if self.config.norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError(
                "norm_type must be one of [rmsnorm, layernorm]. Got "
                f"{self.config.norm_type}.")