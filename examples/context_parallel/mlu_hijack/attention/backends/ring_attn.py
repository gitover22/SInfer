from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from bangtransformer.torch import bt_ops
from sinfer.attention.backends.abstract import AttentionMetadata
from sinfer.attention.backends.flash_attn import FlashAttentionMetadata
from sinfer.attention.ops.paged_attn import PagedAttention
from ...distributed.parallel_state import get_context_model_parallel_group
from ...distributed.ring_comm import RingComm


# code references: https://github.com/zhuzilin/ring-flash-attention
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


def get_half(pack_tensor, cu_seq_lens, first_half):
    batch_num = cu_seq_lens.shape[0] - 1
    half_list = []
    for batch in range(batch_num):
        if first_half:
            start = cu_seq_lens[batch]
            end = (cu_seq_lens[batch] + cu_seq_lens[batch + 1]) // 2
        else:
            start = (cu_seq_lens[batch] + cu_seq_lens[batch + 1]) // 2
            end = cu_seq_lens[batch + 1]
        half = pack_tensor[start: end]
        half_list.append(half)
    half = torch.cat(half_list, dim=0)
    return half


def update_half(pack_tensor, half_tensor, cu_seq_lens, first_half):
    half_cu_seq_lens = cu_seq_lens // 2
    batch_num = cu_seq_lens.shape[0] - 1
    for batch in range(batch_num):
        if first_half:
            start = cu_seq_lens[batch]
            end = (cu_seq_lens[batch] + cu_seq_lens[batch + 1]) // 2
        else:
            start = (cu_seq_lens[batch] + cu_seq_lens[batch + 1]) // 2
            end = cu_seq_lens[batch + 1]
        pack_tensor[start: end] = half_tensor[half_cu_seq_lens[batch]: half_cu_seq_lens[batch + 1]]
        

def zigzag_ring_attn(self,
        query: torch.Tensor, # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_heads. head_size]
        value: torch.Tensor, # [num_tokens, num_heads, head_size]
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata[FlashAttentionMetadata]) -> torch.Tensor:
    num_tokens, _, _ = query.shape
    cu_seq_lens = attn_metadata.prefill_metadata.seq_start_loc
    batch_num = cu_seq_lens.shape[0] - 1
    block_seq_len = query.shape[0] // 2
    comm = RingComm(get_context_model_parallel_group())  # k
    comm_ = RingComm(get_context_model_parallel_group())  # v
    comm__ = RingComm(get_context_model_parallel_group())  # slot_mapping
    
    q, k, v = query, key, value
    if batch_num == 1:
        q1 = q[block_seq_len:]
    else:
        q1 = get_half(q, cu_seq_lens, False)
    slot_mapping = attn_metadata.slot_mapping
    
    out = None
    lse = None
    next_k, next_v = None, None
    next_slot_mapping = None
    
    def forward(q, k, v, causal):
        if batch_num == 1:
            seq = q.shape[0]
            seq_k = k.shape[0]
            cu_seq_lens_q =  torch.arange(0, seq+1, seq, dtype=torch.int32, device="cuda")
            cu_seq_lens_kv = torch.arange(0, seq_k+1, seq_k, dtype=torch.int32, device="cuda")
            max_seq_len_q = seq
            max_seq_len_kv = seq_k
        else:
            max_seq_len_q = attn_metadata.prefill_metadata.max_seq_len
            max_seq_len_kv = attn_metadata.prefill_metadata.max_seq_len
            cu_seq_lens_q = cu_seq_lens
            cu_seq_lens_kv = cu_seq_lens
            if q.shape[0] != cu_seq_lens[-1]:
                cu_seq_lens_q = cu_seq_lens // 2
                max_seq_len_q = max_seq_len_q // 2
            if k.shape[0] != cu_seq_lens[-1]:
                cu_seq_lens_kv = cu_seq_lens // 2
                max_seq_len_kv = max_seq_len_kv // 2
        alibi_slopes = None if self.alibi_slopes is None else \
                                self.alibi_slopes.repeat(attn_metadata.num_prefills, 1)
        ouptuts = bt_ops.flash_attention(q,
                               k,
                               v,
                               cu_seq_lens_q,
                               cu_seq_lens_kv,
                               alibi_slopes,
                               None,
                               max_seq_len_q,
                               max_seq_len_kv,
                               self.scale,
                               causal, -1, -1, torch.float, True)
        block_out, block_lse = ouptuts[0], ouptuts[1]
 
        if block_lse.shape[0] == 1:
            block_lse = block_lse[0]
        else:
            # block_lse shape is [batch, head_num_q, max_seq_q]， the empty part will set 0
            # we need to modify the shape to [batch, head_num_q, total_seq_q]
            block_lse_list = []
            for batch in range(block_lse.shape[0]):
                block_lse_ = block_lse[batch][:, : cu_seq_lens_q[batch + 1] - cu_seq_lens_q[batch]]
                block_lse_list.append(block_lse_)
            block_lse = torch.cat(block_lse_list, dim=-1)
        
        return block_out, block_lse
    
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k.contiguous())
            next_v: torch.Tensor = comm_.send_recv(v.contiguous())
            next_slot_mapping: torch.Tensor = comm__.send_recv(slot_mapping)
            comm.commit()
            comm_.commit()
            comm__.commit()
 
        # call bt_ops.reshape_paged_cache
        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
            bt_ops.reshape_paged_cache(k,
                                       v,
                                       key_cache,
                                       value_cache,
                                       slot_mapping.flatten())

        if step == 0:
            block_out, block_lse = forward(q, k, v, causal = True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            if batch_num == 1:
                k0 = k[:block_seq_len]
                v0 = v[:block_seq_len]
            else:
                k0 = get_half(k, cu_seq_lens, True)
                v0 = get_half(v, cu_seq_lens, True)
            block_out, block_lse = forward(q, k0, v0, causal = False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, causal = False)
            if batch_num == 1:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse,
                                              slice_=(slice(block_seq_len, None)),)
            else:
                slice_out = get_half(out, cu_seq_lens, False)
                slice_lse = get_half(lse, cu_seq_lens, False)
                slice_out, slice_lse = update_out_and_lse(
                    slice_out, slice_lse, block_out, block_lse
                )
                update_half(out, slice_out, cu_seq_lens, False)
                update_half(lse, slice_lse, cu_seq_lens, False)
    
        if step + 1 != comm.world_size:
            comm.wait()
            comm_.wait()
            comm__.wait()
            k = next_k
            v = next_v
            slot_mapping = next_slot_mapping
    out = out.to(q.dtype)
    return out.view(num_tokens, self.num_heads * self.head_size)
