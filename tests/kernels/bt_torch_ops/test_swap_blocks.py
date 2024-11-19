import random

import pytest
import torch
import torch_mlu
USE_CUDA=False
USE_MLU=True
if USE_CUDA:
      from sinfer._C import cache_ops
if USE_MLU:
    from bangtransformer.torch import bt_ops

from typing import List, Tuple

DTYPES = [torch.half, torch.float]
if "3" not in torch.mlu.get_device_name(0):
    DTYPES = [torch.half, torch.bfloat16, torch.float]
SEEDS = [0]
DEVICES = [i for i in range(1 if torch.mlu.device_count() == 1 else 2)]
num_N = [3600]
num_C = [8]
num_H = [64,128]
num_W = [16]
num_pairs = [6,512]
cpys  = ["mlu to mlu", "mlu to cpu", "cpu to mlu"]

class SwapBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                dst: torch.Tensor,
                src: torch.Tensor,
                src_to_dst: dict):
        for key, value in src_to_dst.items():
            dst[value] = src[key]

@pytest.mark.parametrize("n", num_N)
@pytest.mark.parametrize("c", num_C)
@pytest.mark.parametrize("h", num_H)
@pytest.mark.parametrize("w", num_W)
@pytest.mark.parametrize("num_pair", num_pairs)
@pytest.mark.parametrize("cpy", cpys)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_copy_blocks(
    n,
    c: int,
    h: int,
    w: int,
    num_pair: int,
    cpy: str,
    dtype: torch.dtype,
    seed: int,
    device: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.mlu.manual_seed(seed)
    
    if cpy == "mlu to mlu":
        src = torch.randn(n, c, h, w, dtype=dtype).mlu()
        dst = torch.randn(n, c, h, w, dtype=dtype).mlu()
    elif cpy == "mlu to cpu":
        src = torch.randn(n, c, h, w, dtype=dtype).mlu()
        dst = torch.randn(n, c, h, w, dtype=dtype).cpu()
    elif cpy == "cpu to mlu":
        src = torch.randn(n, c, h, w, dtype=dtype).cpu()
        dst = torch.randn(n, c, h, w, dtype=dtype).mlu()
    else:
        print("unkown copy direction.")
        exit(1)

    values = list(range(num_pair))
    random.shuffle(values)
    src_to_dst = {key: value for key, value in zip(range(num_pair), values)}

    ref_src, ref_dst = src.clone(), dst.clone()
    swap_blocks = SwapBlocks()
    # Call the swap blocks kernel.
    # cpu
    swap_blocks(ref_dst, ref_src, src_to_dst)
    # mlu
    bt_ops.swap_blocks(dst, src, src_to_dst)
    # diff
    assert torch.allclose(src, ref_src)
    assert torch.allclose(dst, ref_dst)
