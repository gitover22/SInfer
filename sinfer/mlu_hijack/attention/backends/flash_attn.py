from typing import Optional, Type, List
import torch
import sinfer
from sinfer.logger import init_logger
from sinfer.mlu_hijack.mlu_definitions import USE_PAGED
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from bangtransformer.torch import bt_ops
from sinfer.attention.backends.abstract import AttentionMetadata
from sinfer.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from sinfer.attention.ops.paged_attn import PagedAttention

logger = init_logger(__name__)


sinfer__attention__backends__flash_attn__FlashAttentionImpl____init__org = FlashAttentionImpl.__init__

def sinfer__attention__backends__flash_attn__FlashAttentionImpl____init__(
    self,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: Optional[int] = None,
    alibi_slopes: Optional[List[float]] = None,
    sliding_window: Optional[int] = None,
) -> None:
    sinfer__attention__backends__flash_attn__FlashAttentionImpl____init__org(
            self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window)
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: move alibi_slopes to device
    '''
    if self.alibi_slopes is not None:
        self.alibi_slopes = self.alibi_slopes.cuda()
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def sinfer__attention__backends__flash_attn__FlashAttentionImpl__forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata[FlashAttentionMetadata],
    kv_scale: float,
) -> torch.Tensor:
    """Forward pass with FlashAttention and PagedAttention.

    Args:
        query: shape = [num_tokens, num_heads * head_size]
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads * head_size]
        kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
        attn_metadata: Metadata for attention.
    Returns:
        shape = [num_tokens, num_heads * head_size]
    """
    num_tokens, hidden_size = query.shape
    # Reshape the query, key, and value tensors.
    query = query.view(-1, self.num_heads, self.head_size)
    key = key.view(-1, self.num_kv_heads, self.head_size)
    value = value.view(-1, self.num_kv_heads, self.head_size)
    if kv_cache is not None:
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size)

        # Reshape the input keys and values and store them in the cache.
        # If kv_cache is not provided, the new key and value tensors are
        # not cached. This happens during the initial memory profiling run.
        '''
        =============================
        Modify by sinfer_mlu
        =============================
        @brief: use bt_ops.reshape_paged_cache
        '''
        if USE_PAGED:
            bt_ops.reshape_paged_cache(key,
                                       value,
                                       key_cache,
                                       value_cache,
                                       attn_metadata.slot_mapping.flatten())
        else:
            bt_ops.reshape_linear_cache(key,
                                        value,
                                        key_cache,
                                        value_cache,
                                        attn_metadata.cu_seq_lens,
                                        attn_metadata.max_seq_len,
                                        True, # packed
                                        None, # context_seq_offset
                                        attn_metadata.batch_ids,
                                        attn_metadata.slot_mapping_new)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    num_prefill_tokens = attn_metadata.num_prefill_tokens
    num_decode_tokens = attn_metadata.num_decode_tokens
    assert key.shape[0] == num_prefill_tokens + num_decode_tokens
    assert value.shape[0] == num_prefill_tokens + num_decode_tokens

    output = torch.empty_like(query)
    # Query for decode. KV is not needed because it is already cached.
    decode_query = query[num_prefill_tokens:]
    # QKV for prefill.
    query = query[:num_prefill_tokens]
    key = key[:num_prefill_tokens]
    value = value[:num_prefill_tokens]

    assert query.shape[0] == num_prefill_tokens
    assert decode_query.shape[0] == num_decode_tokens
    if prefill_meta := attn_metadata.prefill_metadata:
        # Prompt run.
        if kv_cache is None or prefill_meta.block_tables.numel() == 0:
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.
            '''
            =============================
            Modify by sinfer_mlu
            =============================
            @brief: use bt_ops.flash_attention
            '''
            alibi_slopes = None if self.alibi_slopes is None else \
                                self.alibi_slopes.repeat(attn_metadata.num_prefills, 1)
            out = bt_ops.flash_attention(query,
                                         key,
                                         value,
                                         prefill_meta.seq_start_loc,
                                         prefill_meta.seq_start_loc,
                                         alibi_slopes,
                                         None,
                                         num_prefill_tokens,
                                         num_prefill_tokens,
                                         self.scale,
                                         True,
                                         -1 if self.sliding_window is None \
                                            else self.sliding_window[0],
                                         -1 if self.sliding_window is None \
                                            else self.sliding_window[1],
                                         torch.float,
                                         False)
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
            assert output[:num_prefill_tokens].shape == out.shape
            output[:num_prefill_tokens] = out
        else:
            # prefix-enabled attention
            # TODO(Hai) this triton kernel has regression issue (broke) to
            # deal with different data types between KV and FP8 KV cache,
            # to be addressed separately.
            '''
            =============================
            Modify by sinfer_mlu
            =============================
            @brief: chunked-prefix use triton
            '''
            x = 16 // key_cache.element_size()
            kc_reshaped = key_cache.view(key_cache.shape[0], key_cache.shape[1],
                    key_cache.shape[2], key_cache.shape[3] // x, x).permute(0, 1, 3, 2, 4)
            kv_reshaped = value_cache.view(value_cache.shape[0], value_cache.shape[1],
                    value_cache.shape[2], value_cache.shape[3]).permute(0, 1, 3, 2)
            output[:num_prefill_tokens] = PagedAttention.forward_prefix(
                query,
                key,
                value,
                kc_reshaped,
                kv_reshaped,
                prefill_meta.block_tables,
                prefill_meta.subquery_start_loc,
                prefill_meta.seq_lens_tensor,
                prefill_meta.context_lens_tensor,
                prefill_meta.max_query_len,
                self.alibi_slopes,
                self.sliding_window[0],
            )
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
    if decode_meta := attn_metadata.decode_metadata:
        # Decoding run.
        '''
        =============================
        Modify by sinfer_mlu
        =============================
        @brief: use bt_ops.single_query_cached_kv_attn
        '''
        alibi_slopes = None if self.alibi_slopes is None \
                            else self.alibi_slopes.repeat(attn_metadata.num_decode_tokens, 1)
        decode_query = decode_query.view(-1, 1, self.num_heads, self.head_size).contiguous()
        out = bt_ops.single_query_cached_kv_attn(decode_query,
                                                 key_cache, value_cache,
                                                 decode_meta.block_tables,
                                                 decode_meta.seq_lens_tensor,
                                                 None, None, # k/v_cache_quant_scale
                                                 alibi_slopes,
                                                 decode_meta.max_seq_len,
                                                 0, 0, # windows_size_left/right
                                                 self.scale)
        output[num_prefill_tokens:] = out.view(-1, self.num_heads, self.head_size)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

    # Reshape the output tensor.
    return output.view(num_tokens, hidden_size)


MluHijackObject.add_hijack_object(FlashAttentionImpl,
                                  FlashAttentionImpl.__init__,
                                  sinfer__attention__backends__flash_attn__FlashAttentionImpl____init__)
MluHijackObject.add_hijack_object(FlashAttentionImpl,
                                  FlashAttentionImpl.forward,
                                  sinfer__attention__backends__flash_attn__FlashAttentionImpl__forward)
