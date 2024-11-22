import numpy as np
import math
import random
import time

import pytest
import torch

from sinfer.mlu_hijack.attention.ops.triton_flash_attention import triton_attention
class SelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale

    # special alibi only support causal
    def build_alibi(self, slopes, block_size, n_heads, dtype):
        device ='mlu'
        tril = torch.tril(torch.ones(1,1 , block_size, block_size, device = device))
        bias_rows = torch.arange( block_size, device=device).view(1, -1)
        bias_cols = torch.arange( block_size, device=device).view(-1, 1)
        bias = - torch.sqrt(bias_cols - bias_rows)
        bias = bias.view(1, block_size, block_size) * slopes.view(-1, 1, 1)
        bias = bias.masked_fill(tril == 0, float('-inf'))
        return bias.type(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cur_seq_len_t:torch.Tensor, alibi_slope:torch.Tensor, attn_bias:torch.Tensor):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D)
            k: The tensor containing the key.   (B, T, H, D)
            v: The tensor containing the value. (B, T, H, D)
            cur_seq_len_t: true_seq_lens. (B+1)
            alibi_slope: (H) or (B, H)
            attn_bias: (B,H,T,T) or (B,T,T)
        """
        batch = q.shape[0]
        seq_q = q.shape[1]
        seq_k = k.shape[1]
        head = q.shape[2]
        scores = torch.einsum('bthd,bshd->bhts', q, k )* self.softmax_scale
        # mask
        if alibi_slope is not None:
            slope = torch.zeros((batch, head)).mlu()
            if len(alibi_slope.shape) == 1 :
                slope[:,]=alibi_slope
            else:
                slope=alibi_slope
            slope = slope.reshape(batch, head, 1, 1)
            slope_bias = torch.zeros(batch, head, seq_q, seq_k).mlu()
            if self.causal:
                relative_pos = torch.arange(-seq_k + 1, 1, dtype=torch.float32).mlu()
                slope_bias = relative_pos * slope
            else:
                row_idx = torch.arange(seq_q, dtype=torch.long).reshape(-1, 1)
                col_idx = torch.arange(seq_k, dtype=torch.long)
                relative_pos = torch.abs(row_idx + seq_k - seq_q - col_idx).mlu()
                slope_bias = -slope * relative_pos.to(dtype=slope.dtype)
            # if use special alibi
            # slope_bias = self.build_alibi(alibi_slope, seq_k, head, dtype=torch.float32)

            scores += slope_bias
        if attn_bias is not None:
            if len(attn_bias.shape) == 3:
                scores += attn_bias.unsqueeze(1)
            else:
                scores +=attn_bias
        if self.causal:
            causal_mask = torch.triu(torch.full((seq_q, seq_k), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        else: # fill -inf in pad_area
            for b in range(batch):
                true_seq_len = cur_seq_len_t[b + 1] - cur_seq_len_t[b]
                scores[b, ..., true_seq_len:] = -10000.0
                scores[b, :, true_seq_len:, :] = -10000.0
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        output = torch.einsum('bhts,bshd->bthd', attention, v)
        return output.contiguous()


NUM_HEADS = [64, 256]
NUM_QUERIES_PER_KV = [1]
HEAD_SIZES = [96]
DTYPES = [torch.float16, torch.float32]

@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_contexted_kv_attention(num_heads: int, num_queries_per_kv: int,
    head_size: int, dtype: torch.dtype) -> None:
    """
    split test case head_size 96 cause multi tests in one pytest will conflict memory.
    """
    device="cuda"

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    random.seed(1)
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads

    num_tokens = sum(query_lens)
    max_seqlens_q = max(query_lens)
    max_seqlens_k = max(query_lens)
    cu_seqlens = [0]
    for value in query_lens:
        cu_seqlens.append(cu_seqlens[-1] + value)
    cu_seqlens_q =  torch.tensor(cu_seqlens, dtype=torch.int, device=device)
    cu_seqlens_k =  torch.tensor(cu_seqlens, dtype=torch.int, device=device)

    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype,device=device)
    query.uniform_(-1e-3, 1e-3)
    triton_output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype,device=device)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype,device=device)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size,
                          block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    v_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size,
                          dtype=dtype, device=device)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype,device=device)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype,device=device)
    values = torch.arange(0, cache_size, dtype=torch.long,device=device)
    values = values[torch.randperm(cache_size,device=device)]
    block_table = values[:BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long,device=device)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long,device=device)
    b_start_loc = torch.cumsum(torch.tensor(
                               [0] + query_lens[:-1], dtype=torch.long,device=device), dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                   dtype=torch.long,device=device), dim=0)
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8,
                           8).permute(0, 2, 3, 1, 4).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_kv_heads,
                           head_size).permute(0, 2, 3, 1).contiguous()

    triton_output,_ = triton_attention(query,
          k,
          v,
          None,
          cu_seqlens_q,
          cu_seqlens_k,
          max_seqlens_q,
          max_seqlens_k)
    triton_output_cpu = triton_output.to(device='cpu')
    def copy_pack_data_to_pad_data(pad_input: torch.Tensor,
                                   packed_input_list: list,
                                   t_len_sequence: list,
                                   max_sequence_len: int):
        end_index1 = 0
        for index in range(len(t_len_sequence)):
            start_index1 = end_index1
            end_index1 = end_index1 + t_len_sequence[index]
            start_index = index * max_sequence_len
            end_index = start_index + t_len_sequence[index]
            pad_input[start_index:end_index, ...] = packed_input_list[start_index1:end_index1, ...]

    pad_input_q = torch.zeros((MAX_SEQ_LEN * BS, num_heads, head_size)).mlu().half()
    pad_input_k = torch.zeros((MAX_SEQ_LEN * BS, num_kv_heads, head_size)).mlu().half()
    pad_input_v = torch.zeros((MAX_SEQ_LEN * BS, num_kv_heads, head_size)).mlu().half()
    copy_pack_data_to_pad_data(pad_input_q, query, query_lens, MAX_SEQ_LEN)
    copy_pack_data_to_pad_data(pad_input_k, k,     query_lens, MAX_SEQ_LEN)
    copy_pack_data_to_pad_data(pad_input_v, v,     query_lens, MAX_SEQ_LEN)
    softmax_scale = 1 / math.sqrt(head_size)
    attention = SelfAttention(causal = False, softmax_scale=softmax_scale)
    torch_output = attention(pad_input_q.view(BS, MAX_SEQ_LEN, num_heads, head_size),
                             pad_input_k.view(BS, MAX_SEQ_LEN, num_kv_heads, head_size),
                             pad_input_v.view(BS, MAX_SEQ_LEN, num_kv_heads, head_size),
                             cu_seqlens_q,None,None)
    pad_triton_output = torch_output.clone().view(BS * MAX_SEQ_LEN, num_heads, head_size)
    copy_pack_data_to_pad_data(pad_triton_output, triton_output_cpu, query_lens, MAX_SEQ_LEN)
    view_triton_output = pad_triton_output.view(BS, MAX_SEQ_LEN, num_heads, head_size)
    torch.testing.assert_close(view_triton_output, torch_output)


HEAD_SIZES = [24, 128]
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_contexted_kv_attention_1(num_heads: int, num_queries_per_kv: int, head_size: int,
    dtype: torch.dtype) -> None:
    """
    split test case ihead_size 24, 128 cause multi tests in one pytest will conflict memory.
    """
    test_contexted_kv_attention(num_heads, num_queries_per_kv, head_size, dtype)
