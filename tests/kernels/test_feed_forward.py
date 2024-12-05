import pytest
import numpy
import torch
from sinfer.config import ParallelConfig
from sinfer.model_executor.model_loader.weight_utils import default_weight_loader, initialize_dummy_weights
from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward
from sinfer.model_executor.model_loader.utils import set_default_torch_dtype
from sinfer.model_executor.utils import set_random_seed
from sinfer.test_utils import init_test_distributed_environment


def compute_diff(baseline: numpy.ndarray, compare: numpy.ndarray):
    error = numpy.abs(baseline - compare)
    diff1 = numpy.sum(error) / numpy.sum(numpy.abs(baseline))
    diff2 = numpy.sqrt(numpy.sum(error**2)/numpy.sum(baseline**2))
    return diff1, diff2


BATCH_SIZE = [1]
SEQ_LENS = [128]
HIDDEN_SIZE = [32]
INTERMEDIATE_SIZE = [64]
HIDDEN_ACT = ['silu', 'gelu']
IS_GATED = [True, False]
BIAS = [True, False]
UP_PROJ_NAME = ['up_proj']
DOWN_PROJ_NAME = ['down_proj']
DTYPE = [torch.float16]
SEED = [0]

@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZE)
@pytest.mark.parametrize("hidden_act", HIDDEN_ACT)
@pytest.mark.parametrize("is_gated", IS_GATED)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("up_proj_name", UP_PROJ_NAME)
@pytest.mark.parametrize("down_proj_name", DOWN_PROJ_NAME)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("seed", SEED)
@torch.inference_mode()
def test_feed_forward(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
    is_gated: bool,
    bias: bool,
    up_proj_name: str,
    down_proj_name: str,
    dtype: torch.dtype,
    seed : int
) -> None:
    device = torch.device("mlu:0")
    set_random_seed(seed)

    # init distributed environment
    # now only support tensor_parallel_size=1 and pipeline_parallel_size=1
    if not torch.distributed.is_initialized():
        init_test_distributed_environment(pipeline_parallel_size=1,
                                    tensor_parallel_size=1,
                                    rank=0,
                                    distributed_init_port="3000",
                                    local_rank=0)

    with set_default_torch_dtype(dtype):
        # create ffn and initialize weights
        ffn = FeedForward(hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        hidden_act=hidden_act,
                        up_proj_name=up_proj_name,
                        is_gated=is_gated,
                        down_proj_name=down_proj_name,
                        bias=bias).to(device)
        initialize_dummy_weights(ffn, low=-1e-1, high=1e-1)

    # create input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    # ffn forward
    out = ffn(hidden_states)
    # reference ffn forward
    ref_out = ffn._forward(hidden_states)

    # compute the diff1 and diff2 value, for fp16, the threshold is 5e-3
    diff1, diff2 = compute_diff(baseline=ref_out.cpu().float().detach().numpy(),
                                compare=out.cpu().float().detach().numpy())

    del ffn, hidden_states, out, ref_out

    assert diff1 <= 5e-3 and diff2 <= 5e-3
