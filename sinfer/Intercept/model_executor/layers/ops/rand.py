from typing import Optional, Union

import torch
import triton
import triton.language as tl

import sinfer
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject

def sinfer__model_executor__layers__ops__rand__seeded_uniform(
    *size,
    seeds: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
    pin_memory: Optional[bool] = False,
) -> torch.Tensor:
    """Similar to torch.rand, but allows for seeds to be set per row.

    seeds must be a 1d tensor. The output tensor may be 1d, 2d, or 3d.
    If it is 3d, the additional seeds needed will be derived automatically
    in a deterministic fashion:
    [
        row 0: [columns_with_seed_0], [columns_with_seed0^1], ...
    ]
    """
    n_dims = len(size)
    if n_dims > 3:
        raise ValueError("seeded_uniform only supports up to 3D tensors")

    if out is None:
        out = torch.empty(*size,
                          dtype=dtype,
                          device=device,
                          pin_memory=pin_memory)
    elif out.shape != size:
        raise ValueError("shape of out and size must be the same")

    if n_dims == 3:
        n_rows, n_3d, n_cols = out.shape
        stride_row = out.stride(0)
        stride_3d = out.stride(1)
    elif n_dims == 2:
        n_rows, n_cols = out.shape
        n_3d = 1
        stride_row = out.stride(0)
        stride_3d = 1
    else:
        n_cols = out.shape[0]
        n_rows = 1
        n_3d = 1
        stride_row = 1
        stride_3d = 1

    if seeds.ndim != 1:
        raise ValueError("seeds must be a 1D tensor")

    if seeds.numel() != n_rows:
        raise ValueError(
            "seeds must have the same number of elements as out has rows")

    # The philox PRNG Triton uses generates 4 random numbers at once.
    # Therefore, the most efficient use of it is to divide the
    # block size by 4, and then save the generated random numbers to
    # each of the 4 slices of the tensor.
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: disable custom_all_reduce, re-set block_size to support paged and unpaged mode
    '''
    sinfer__model_executor__layers__ops__rand__seeded_uniform_triton[(n_rows, n_3d, 1)](
        out,
        seeds,
        stride_row,
        stride_3d,
        seeds.stride(0),
        n_rows,
        n_3d,
        n_cols,
    )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return out

@triton.jit
def sinfer__model_executor__layers__ops__rand__seeded_uniform_triton(
    out_ptr: torch.Tensor,
    seed_ptr: torch.Tensor,
    out_row_stride: int,
    out_3d_stride: int,
    seed_row_stride: int,
    n_rows: int,
    n_3d: int,
    n_cols: int,
):
    """
    Generate a random float32 number in [0, 1) for each element in the output
    tensor. The random numbers in a row generated using the seed for that row.

    Args:
        out_ptr: The output tensor.
        seed_ptr: The per-row seeds to use for random number generation.
        out_row_stride: The stride between rows of the output tensor.
        out_3d_stride: The stride between 3D slices of the output tensor.
        seed_row_stride: The stride between rows of the seed tensor.
        n_rows: The number of rows in the output tensor.
        n_3d: The size of second dimension of the output tensor,
            if output tensor is 3D.
        n_cols: The number of columns in the output tensor.
        n_slices: The number of philox outputs to use.
    """
    # Get the row index.
    row_idx = tl.program_id(axis=0)
    three_d_idx = tl.program_id(axis=1)
    """
    ===================================
    Add by sinfer_mlu
      cause mlu RAM size limit, add split before rand4x，
      The previous logic was:
      get a cols, and rand4x return it.
      The modified logic is:
      get a cols, slice it by SLICE_SIZE(4096), then execute rand4x for each slice.
    ===================================
    """
    SLICE_SIZE = 4096
    slice_for = n_cols // SLICE_SIZE + (1 if n_cols % SLICE_SIZE else 0)

    seed = tl.load(seed_ptr + row_idx * seed_row_stride)
    if three_d_idx > 0:
        seed ^= three_d_idx
    for slice_i in range(slice_for):
        slice_offset = slice_i * SLICE_SIZE
        philox_offsets = slice_offset + tl.arange(0, 1024)
        # Get the seed for the current element.
        # Generate random numbers in [0, 1).
        out1, out2, out3, out4 = tl.rand4x(seed, philox_offsets)
        output_row_start_ptr = (out_ptr + row_idx * out_row_stride +
                                three_d_idx * out_3d_stride)
        out1_offsets =          philox_offsets
        out2_offsets = 1*1024 + philox_offsets
        out3_offsets = 2*1024 + philox_offsets
        out4_offsets = 3*1024 + philox_offsets
        tl.store(output_row_start_ptr + philox_offsets,
                 out1,
                 mask=out1_offsets < n_cols)
        if n_cols - slice_offset > 1024:
            tl.store(output_row_start_ptr + out2_offsets,
                     out2,
                     mask=out2_offsets < n_cols)
        if n_cols - slice_offset > 2048:
            tl.store(output_row_start_ptr + out3_offsets,
                     out3,
                     mask=out3_offsets < n_cols)
        if n_cols - slice_offset > 3072:
            tl.store(output_row_start_ptr + out4_offsets,
                     out4,
                     mask=out4_offsets < n_cols)
    """
    ===================================
    End of MLU Hijack
    ===================================
    """

MluHijackObject.add_hijack_object(sinfer.model_executor.layers.ops.rand,
                                  sinfer.model_executor.layers.ops.rand.seeded_uniform,
                                  sinfer__model_executor__layers__ops__rand__seeded_uniform)

MluHijackObject.add_hijack_object(sinfer.model_executor.layers.ops.sample,
                                  sinfer.model_executor.layers.ops.sample.seeded_uniform,
                                  sinfer__model_executor__layers__ops__rand__seeded_uniform)
