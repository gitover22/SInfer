from typing import Tuple, List, Dict
import torch
import sinfer
from sinfer.logger import init_logger
from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from bangtransformer.torch import bt_ops
from sinfer.attention.ops.paged_attn import PagedAttention

logger = init_logger(__name__)

@staticmethod
def intercept_split_kv_cache(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # fit bt_ops.reshape_paged_cache shape requirements
    num_blocks = kv_cache.shape[1]
    key_cache = kv_cache[0]
    key_cache = key_cache.view(num_blocks, num_kv_heads, -1, head_size)
    value_cache = kv_cache[1]
    value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)

    return key_cache, value_cache


@staticmethod
def intercept_copy_blocks(
    kv_caches: List[torch.Tensor],
    src_to_dists: Dict[int, List[int]],
) -> None:
    key_caches = [kv_cache[0] for kv_cache in kv_caches]
    value_caches = [kv_cache[1] for kv_cache in kv_caches]
    # use bt_ops.copy_blocks
    bt_ops.copy_blocks(key_caches, value_caches, src_to_dists)



@staticmethod
def intercept_swap_blocks(
    src_kv_cache: torch.Tensor,
    dst_kv_cache: torch.Tensor,
    src_to_dst: Dict[int, int],
) -> None:
    # use bt_ops.swap_blocks
    src_key_cache = src_kv_cache[0]
    dst_key_cache = dst_kv_cache[0]
    bt_ops.swap_blocks(dst_key_cache, src_key_cache, src_to_dst)

    src_value_cache = src_kv_cache[1]
    dst_value_cache = dst_kv_cache[1]
    bt_ops.swap_blocks(dst_value_cache, src_value_cache, src_to_dst)

InterceptObject.add_intercept(PagedAttention,
                                  PagedAttention.split_kv_cache,
                                  intercept_split_kv_cache)
InterceptObject.add_intercept(PagedAttention,
                                  PagedAttention.copy_blocks,
                                  intercept_copy_blocks)
InterceptObject.add_intercept(PagedAttention,
                                  PagedAttention.swap_blocks,
                                  intercept_swap_blocks)