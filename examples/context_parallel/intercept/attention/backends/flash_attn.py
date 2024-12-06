from typing import Optional, Type, List
import torch
import sinfer
from sinfer.Intercept.mlu_definitions import USE_PAGED
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.attention.backends.abstract import AttentionMetadata
from sinfer.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from sinfer.attention.ops.paged_attn import PagedAttention
from .ring_attn import zigzag_ring_attn
from sinfer.Intercept.attention.backends.flash_attn import intercept_FlashAttentionImpl__forward
from ...distributed.parallel_state import get_context_model_parallel_world_size


def sinfer__attention__backends__flash_attn__FlashAttentionImpl__forward_wraper(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata[FlashAttentionMetadata],
    kv_scale: float,
) -> torch.Tensor:
    '''
    @brief: use ring attn when context parallel
    '''
    if get_context_model_parallel_world_size() > 1 and attn_metadata.prefill_metadata:
        return zigzag_ring_attn(self,
                                query=query.view(-1, self.num_heads, self.head_size),
                                key=key.view(-1, self.num_kv_heads, self.head_size),
                                value=value.view(-1, self.num_kv_heads, self.head_size),
                                kv_cache=kv_cache,
                                attn_metadata=attn_metadata)
    '''
    =======================
    '''
    return intercept_FlashAttentionImpl__forward(self, query, key, value, kv_cache, attn_metadata, kv_scale)
    
    
InterceptObject.add_intercept(FlashAttentionImpl,
                                  FlashAttentionImpl.forward,
                                  sinfer__attention__backends__flash_attn__FlashAttentionImpl__forward_wraper)