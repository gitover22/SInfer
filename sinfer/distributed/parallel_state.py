# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Tensor and pipeline parallel groups."""
import contextlib
from typing import Optional

import torch

import sinfer.envs as envs
from sinfer.logger import init_logger

logger = init_logger(__name__)

# 当前 rank 所属的张量模型并行组 TP
_TP_DEVICE_GROUP = None
_TP_CPU_GROUP = None
# 当前 rank 所属的流水线模型并行组 PP
_PIPELINE_MODEL_PARALLEL_GROUP = None

# 当用户直接调用 `torch.distributed.all_reduce` 等操作时，
# 会默认使用这个分组。该分组通过 `init_distributed_environment` 函数中的
# `backend` 参数初始化。实际上，这个分组就是 `torch.distributed.group.WORLD`。
# 这里特别注明，这个变量是设备相关的。
# 注意，这个变量并不安全，因为如果用户先调用了 `init_distributed_environment`
# 然后又手动销毁了进程组，这个变量仍然会引用已被销毁的进程组，导致无效
_DEVICE_WORLD_GROUP = None

# 在调用 init_distributed_environment 时，我们还会初始化一个使用 `gloo` 后端的分组，用于通过 CPU 在进程之间直接协调。
_CPU_WORLD_GROUP = None

# 总结：在调用 `init_distributed_environment` 后，
# 我们始终会有两个分组：一个是设备相关的（默认分组），另一个是 CPU 分组。
# 所有的进程都会属于这两个分组。

# 每个流水线分组的全局 rank 列表，用于在广播操作中
# 方便地计算来自流水线第一阶段或最后阶段的源 rank
_PIPELINE_GLOBAL_RANKS = None

_LOCAL_RANK = -1



def get_local_rank():
    global _LOCAL_RANK
    return _LOCAL_RANK

# 初始化分布式环境
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d "
        "distributed_init_method=%s backend=%s", world_size, rank, local_rank,
        distributed_init_method, backend)
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment")
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank)
        global _DEVICE_WORLD_GROUP, _CPU_WORLD_GROUP
        _DEVICE_WORLD_GROUP = torch.distributed.group.WORLD
        ranks = list(range(torch.distributed.get_world_size()))
        _CPU_WORLD_GROUP = torch.distributed.new_group(ranks=ranks,
                                                       backend="gloo")
        # set the local rank
        # local_rank is not available in torch ProcessGroup,
        # see https://github.com/pytorch/pytorch/issues/122816
        if local_rank == -1 and distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        global _LOCAL_RANK
        _LOCAL_RANK = local_rank

# 模型并行初始化
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    初始化模型并行组。

    参数:

    - tensor_model_parallel_size: 用于张量模型并行的GPU数量。
    - pipeline_model_parallel_size: 用于流水线模型并行的GPU数量。

    示例解释:

    假设总共有8个GPU, 标记为g0到g7, 使用2个GPU进行张量模型并行, 4个GPU进行流水线模型并行。
    这个函数将创建以下并行组：
    1. 张量模型并行组 4个组: [g0, g1] [g2, g3] [g4, g5] [g6, g7]
    2. 流水线模型并行组 2个组: [g0, g2, g4, g6] [g1, g3, g5, g7]

    注意事项
    为了提高效率, 要确保相邻的GPU排名位于同一个DGX服务器中。例如使用2台DGX-1服务器(每台有8个GPU) ,则排名0到7属于第一台服务器, 排名8到15属于第二台服务器
    这能够确保通信效率最优，并最大化分布式训练的性能优势。
    """

    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend() # cncl

    if (world_size !=
            tensor_model_parallel_size * pipeline_model_parallel_size):
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

    num_tensor_model_parallel_groups: int = (world_size //
                                             tensor_model_parallel_size) # 计算张量并行组的数量
    num_pipeline_model_parallel_groups: int = (world_size //
                                               pipeline_model_parallel_size) # 计算流水线并行组的数量
    rank = torch.distributed.get_rank()

    # 创建张量模型并行组
    global _TP_DEVICE_GROUP, _TP_CPU_GROUP
    assert _TP_DEVICE_GROUP is None, (
        "tensor model parallel group is already initialized")
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks, backend=backend)
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            _TP_DEVICE_GROUP = group
            _TP_CPU_GROUP = cpu_group

    # 创建流水线模型并行组
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, (
        "pipeline model parallel group is already initialized")
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks, backend=backend)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks

# 确保并行组已初始化
def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """辅助函数，用于在模型并行组尚未初始化时进行初始化；
    或者在模型并行组已经初始化的情况下，确保张量并行和流水线并行的大小与预期值相等。
    """
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend() # cncl
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size,
                                  pipeline_model_parallel_size, backend)
        return

    assert (
        get_tensor_model_parallel_world_size() == tensor_model_parallel_size
    ), ("tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}")
    assert (get_pipeline_model_parallel_world_size(
    ) == pipeline_model_parallel_size), (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{get_pipeline_model_parallel_world_size()=} vs. "
        f"{pipeline_model_parallel_size=}")


def model_parallel_is_initialized():
    """检查张量并行组和流水线并行组是否已初始化。"""
    return (_TP_DEVICE_GROUP is not None
            and _PIPELINE_MODEL_PARALLEL_GROUP is not None)


def get_cpu_world_group():
    """Get the CPU world group."""
    assert _CPU_WORLD_GROUP is not None, ("CPU world group is not initialized")
    return _CPU_WORLD_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TP_DEVICE_GROUP is not None, (
        "tensor model parallel group is not initialized")
    return _TP_DEVICE_GROUP


def get_tensor_model_parallel_cpu_group():
    """Get the tensor model parallel cpu group the caller rank belongs to."""
    assert _TP_CPU_GROUP is not None, (
        "tensor model parallel cpu group is not initialized")
    return _TP_CPU_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, (
        "pipeline model parallel group is not initialized")
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(
        group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return torch.distributed.get_world_size(
        group=get_pipeline_model_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return torch.distributed.get_rank(
        group=get_pipeline_model_parallel_group())


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that precedes the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP_DEVICE_GROUP
    if _TP_DEVICE_GROUP:
        torch.distributed.destroy_process_group(_TP_DEVICE_GROUP)
    _TP_DEVICE_GROUP = None
    global _TP_CPU_GROUP
    if _TP_CPU_GROUP:
        torch.distributed.destroy_process_group(_TP_CPU_GROUP)
    _TP_CPU_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    if _PIPELINE_MODEL_PARALLEL_GROUP:
        torch.distributed.destroy_process_group(_PIPELINE_MODEL_PARALLEL_GROUP)
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None
    from sinfer.distributed.device_communicators import pynccl_utils

    # Destroy the pynccl states if any.
    pynccl_utils.destroy_process_group()


# 在使用 CUDA 图时，使用 PyNCCL 进行 all reduce 操作，因为 torch.distributed 对 CUDA 图的支持并不完善
_ENABLE_PYNCCL_FOR_ALL_REDUCE = False


@contextlib.contextmanager
def with_pynccl_for_all_reduce():
    from sinfer.distributed.device_communicators import pynccl_utils
    """use pynccl instead of torch.distributed for all reduce"""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        # don't initialize pynccl when tp_size is 1.
        yield
    else:
        global _ENABLE_PYNCCL_FOR_ALL_REDUCE
        old = _ENABLE_PYNCCL_FOR_ALL_REDUCE
        _ENABLE_PYNCCL_FOR_ALL_REDUCE = True

        stream = torch.cuda.current_stream()
        with pynccl_utils.set_pynccl_stream(stream):
            yield
        _ENABLE_PYNCCL_FOR_ALL_REDUCE = old


def is_pynccl_enabled_for_all_reduce():
    """check if pynccl is enabled for all reduce"""
    global _ENABLE_PYNCCL_FOR_ALL_REDUCE
    return _ENABLE_PYNCCL_FOR_ALL_REDUCE
