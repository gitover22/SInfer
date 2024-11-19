import torch
from sinfer.distributed import parallel_state
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from typing import Optional
from sinfer.distributed.parallel_state import (get_tensor_model_parallel_world_size, 
                                             get_pipeline_model_parallel_world_size)


# Context model parallel group that the current rank belongs to.
_CONTEXT_MODEL_PARALLEL_GROUP = None
_CONTEXT_GLOBAL_RANKS = None


sinfer__distributed__parallel_state__model_parallel_is_initialized_org = parallel_state.model_parallel_is_initialized
sinfer__distributed__parallel_state__model_destroy_model_parallel_org = parallel_state.destroy_model_parallel


def sinfer__distributed__parallel_state__initialize_model_parallel_wraper(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend()

    if (world_size !=
            tensor_model_parallel_size * pipeline_model_parallel_size * context_model_parallel_size):
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size}) x"
            f"context_model_parallel_size ({context_model_parallel_size})")

    num_tensor_model_parallel_groups: int = (world_size //
                                             tensor_model_parallel_size)
    num_pipeline_model_parallel_groups: int = (world_size //
                                               pipeline_model_parallel_size)
    num_context_model_parallel_groups: int = (world_size //
                                              context_model_parallel_size)
    rank = torch.distributed.get_rank()

    # Build the tensor model-parallel groups.
    assert parallel_state._TP_DEVICE_GROUP is None, (
        "tensor model parallel group is already initialized")
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks, backend=backend)
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            parallel_state._TP_DEVICE_GROUP = group
            parallel_state._TP_CPU_GROUP = cpu_group

    # Build the pipeline model-parallel groups.
    assert parallel_state._PIPELINE_MODEL_PARALLEL_GROUP is None, (
        "pipeline model parallel group is already initialized")
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks, backend=backend)
        if rank in ranks:
            parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = group
            parallel_state._PIPELINE_GLOBAL_RANKS = ranks
            
    # Build the context model-parallel groups.
    global _CONTEXT_MODEL_PARALLEL_GROUP
    global _CONTEXT_GLOBAL_RANKS
    assert _CONTEXT_MODEL_PARALLEL_GROUP is None, (
        "context model parallel group is already initialized")
    for i in range(num_context_model_parallel_groups):
        ranks = range(i, context_model_parallel_size * tensor_model_parallel_size + i, tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _CONTEXT_MODEL_PARALLEL_GROUP = group
            _CONTEXT_GLOBAL_RANKS = ranks


def sinfer__distributed__parallel_state__ensure_model_parallel_initialized_wraper(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend()
    
    if not sinfer__distributed__parallel_state__model_parallel_is_initialized_wraper():
        sinfer__distributed__parallel_state__initialize_model_parallel_wraper(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_model_parallel_size,
            backend)
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
    assert (get_context_model_parallel_world_size(
    ) == context_model_parallel_size), (
        "context parallel group already initialized, but of unexpected size: "
        f"{get_context_model_parallel_world_size()=} vs. "
        f"{context_model_parallel_size=}")


def sinfer__distributed__parallel_state__model_parallel_is_initialized_wraper():
    return (sinfer__distributed__parallel_state__model_parallel_is_initialized_org()
            and _CONTEXT_MODEL_PARALLEL_GROUP is not None)


def sinfer__distributed__parallel_state__destroy_model_parallel_wraper():
    global _CONTEXT_MODEL_PARALLEL_GROUP
    if _CONTEXT_MODEL_PARALLEL_GROUP:
        torch.distributed.destroy_process_group(_CONTEXT_MODEL_PARALLEL_GROUP)
    _CONTEXT_MODEL_PARALLEL_GROUP = None
    global _CONTEXT_GLOBAL_RANKS
    _CONTEXT_GLOBAL_RANKS = None
    sinfer__distributed__parallel_state__model_destroy_model_parallel_org()


def get_context_model_parallel_group():
    assert _CONTEXT_MODEL_PARALLEL_GROUP is not None, (
        "context model parallel group is not initialized")
    return _CONTEXT_MODEL_PARALLEL_GROUP


def get_context_model_parallel_world_size():
    return torch.distributed.get_world_size(
        group=get_context_model_parallel_group())
    
    
def get_context_model_parallel_rank():
    return torch.distributed.get_rank(group=get_context_model_parallel_group())


MluHijackObject.add_hijack_object(parallel_state,
                                  "initialize_model_parallel",
                                  sinfer__distributed__parallel_state__initialize_model_parallel_wraper)
MluHijackObject.add_hijack_object(parallel_state,
                                  "model_parallel_is_initialized",
                                  sinfer__distributed__parallel_state__model_parallel_is_initialized_wraper)
MluHijackObject.add_hijack_object(parallel_state,
                                  "destroy_model_parallel",
                                  sinfer__distributed__parallel_state__destroy_model_parallel_wraper)
MluHijackObject.add_hijack_object(parallel_state,
                                  "ensure_model_parallel_initialized",
                                  sinfer__distributed__parallel_state__ensure_model_parallel_initialized_wraper)

