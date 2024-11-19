import pytest
import torch
import torch_mlu
import torch.nn.functional as F
from bangtransformer.torch import bt_ops

def ref_linear(inputs, weights, bias):
    return F.linear(inputs, weights, bias=bias)

BATCH_SIZE = [1,8,64,256]
INPUT_SIZE = [16,41, 128, 577, 857, 996]
OUTPUT_SIZE = [16, 64, 113, 256]
DTYPES = [torch.half, torch.float]
if "3" not in torch.mlu.get_device_name(0):
    DTYPES = [torch.half, torch.bfloat16, torch.float]
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("input_size", INPUT_SIZE)
@pytest.mark.parametrize("output_size", OUTPUT_SIZE)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [0])
def test_attention_project(
    batch_size: int,
    input_size: int,
    output_size: int,
    use_bias: bool,
    dtype: torch.dtype,
    seed : int
) -> None:
    device_id = "mlu:0"
    torch.random.manual_seed(seed)
    torch.mlu.manual_seed(seed)
    input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device=device_id)
    weights = torch.randn(output_size, input_size, dtype=dtype, device=device_id)
    if use_bias:
        bias = torch.randn(output_size, dtype=dtype, device=device_id)
    else:
        bias = None
    out = bt_ops.attention_project(input_tensor, weights, bias)
    ref_out = ref_linear(input_tensor, weights, bias)
    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)


