import random
from typing import List, Optional, Tuple

import pytest
import sinfer
import torch
import torch_mlu
from sinfer.model_executor.layers.rotary_embedding import get_rope

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.half, torch.bfloat16, torch.float] if "3" not in torch.cuda.get_device_name(0) \
    else [torch.half, torch.float]
HEAD_SIZES = [64, 80, 96, 112, 128] #256 failed cause bt bug BT-1017
ROTARY_DIMS = [32]  # None means rotary dim == head size
NUM_HEADS = [9, 17]  # Arbitrary values for testing
BATCH_SIZES = [1, 5]  # Arbitrary values for testing
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
DEVICES = [0]

#TODO:
@pytest.mark.skip(reason="Skipping this test beacause there is no rope._forward_mlu_torch anymore.")
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: int,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size
    torch.random.manual_seed(seed)
    torch.mlu.manual_seed(seed)
    mlu_id = f"mlu:{device}"
    if rotary_dim is None:
        rotary_dim = head_size

    total_seq_len = batch_size * seq_len
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype=dtype, device=mlu_id)

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=mlu_id)

    context_shape = (total_seq_len, num_heads, head_size)
    context = torch.randn(size=context_shape, dtype=dtype).mlu()
    qk = context[..., 0 : num_heads, :]
    max_context_len = seq_len
    cu_seq_lens = torch.arange(0, batch_size + 1, dtype=torch.int32).mlu() * seq_len

    ref_qk_out = torch.empty_like(qk)
    ref_qk = qk.clone()
    qk_out = torch.empty_like(qk)

    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding._set_cu_seq_len(cu_seq_lens)
    rope.cu_seq_len = cu_seq_lens
    qk_out = rope.forward(positions, qk)
    cos_cache = sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.cos_
    sin_cache = sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.sin_
    rope._forward(ref_qk_out,
                  ref_qk.float(),
                  sin_cache.float(),
                  cos_cache.float(),
                  None,
                  cu_seq_lens,
                  is_neox_style,
                  False,
                  False,
                  max_context_len)
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.unset_cos_sin()
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.unset_cu_seq_len()
    assert torch.allclose(ref_qk_out, qk_out, atol=1e-5, rtol=1e-5)
