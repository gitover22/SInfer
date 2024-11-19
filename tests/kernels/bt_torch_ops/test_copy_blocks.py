import random

import pytest
import torch
import torch_mlu
from bangtransformer.torch import bt_ops

from typing import List, Tuple

DTYPES = [torch.half, torch.float]
if "3" not in torch.mlu.get_device_name(0):
    DTYPES = [torch.half, torch.float]
NUM_TOKENS = [83]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256, 512]
BLOCK_SIZES = [8, 16, 32]
NUM_BLOCKS = [1024, 3600]  # Arbitrary values for testing
NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]
DEVICES = [i for i in range(1 if torch.mlu.device_count() == 1 else 2)]


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = head_size**-0.5
    # sinfer scale
    # x = 16 // torch.tensor([], dtype=dtype).element_size()
    # key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_cache_shape = (num_blocks, num_heads, block_size, head_size)
    print("key_cache_shape: ", key_cache_shape)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=dtype).mlu()
        if dtype == torch.int32 or dtype == torch.int64:
            key_cache.random_(-100,100)
        else:
            key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    print("value_cache_shape: ", value_cache_shape)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                dtype=dtype).mlu()
        if dtype == torch.int32 or dtype == torch.int64:
            value_cache.random_(-100,100)
        else:
            value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches

@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_copy_blocks(
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.mlu.manual_seed(seed)
    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 3 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping = {}
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping[src] = [dst1, dst2]

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches(num_blocks, block_size,
                                                num_layers, num_heads,
                                                head_size, dtype, seed)

    # Clone the KV caches.
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    # Call the copy blocks kernel.
    bt_ops.copy_blocks(key_caches, value_caches, block_mapping)

    # Run the reference implementation.
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for cloned_key_cache in cloned_key_caches:
                cloned_key_cache[dst].copy_(cloned_key_cache[src])
            for cloned_value_cache in cloned_value_caches:
                cloned_value_cache[dst].copy_(cloned_value_cache[src])

    # Compare the results.
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches):
        assert torch.allclose(key_cache, cloned_key_cache)
    for value_cache, cloned_value_cache in zip(value_caches,
                                               cloned_value_caches):
        assert torch.allclose(value_cache, cloned_value_cache)
