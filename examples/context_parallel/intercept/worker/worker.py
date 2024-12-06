import os
import torch
import torch.distributed
from sinfer.Intercept.intercept_utils import InterceptObject
from typing import Optional
from sinfer.worker.worker import Worker
from sinfer.worker import worker
from sinfer.config import ParallelConfig
from sinfer.distributed import (get_tensor_model_parallel_cpu_group,
                              init_distributed_environment)
from sinfer.distributed.device_communicators import pynccl_utils
from sinfer.distributed.device_communicators.custom_all_reduce import (
    init_custom_ar)
from sinfer.Intercept.worker.worker import intercept_check_if_gpu_supports_dtype
from sinfer.model_executor import set_random_seed
from ..distributed.parallel_state import sinfer__distributed__parallel_state__ensure_model_parallel_initialized_wraper


def sinfer__worker__worker__Worker__init_device_wraper(self) -> None:
    if self.device_config.device.type == "cuda":
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        '''
        =============================
        Modify by sinfer_mlu
        =============================
        @brief: change *NCCL* environ to *CNCL*, call hijack sinfer__worker__worker___check_if_gpu_supports_dtype
        '''  
        os.environ["TORCH_CNCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("CNCL_ASYNC_ERROR_HANDLING", None)
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        intercept_check_if_gpu_supports_dtype(self.model_config.dtype)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''
        torch.cuda.empty_cache()
        self.init_gpu_memory = torch.cuda.mem_get_info()[0]
    else:
        raise RuntimeError(
            f"Not support device type: {self.device_config.device}")
    # Initialize the distributed environment.
    '''
    ==========================
    Modify by Context Parallel
    ==========================
    @brief: ensure use context parallel hijack init_worker_distributed_environment 
    '''  
    sinfer__worker__worker__init_worker_distributed_environment_wraper(
        self.parallel_config, self.rank,
        self.distributed_init_method,
        self.local_rank)
    '''
    =======================
    End of Context Parallel
    =======================
    '''
    # Set random seed.
    set_random_seed(self.model_config.seed)


def sinfer__worker__worker__init_worker_distributed_environment_wraper(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    '''
    ==========================
    Modify by Context Parallel
    ==========================
    @brief: use context parallel ensure_model_parallel_initialized 
    '''
    sinfer__distributed__parallel_state__ensure_model_parallel_initialized_wraper(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.context_parallel_size)
    '''
    =======================
    End of Context Parallel
    =======================
    '''

    if pynccl_utils.is_initialized():
        pynccl_world_size = pynccl_utils.get_world_size()
        if pynccl_world_size != parallel_config.world_size:
            raise RuntimeError(
                "pynccl is already initialized but the pynccl world "
                "size does not match parallel_config.world_size "
                f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    elif parallel_config.world_size > 1:
        # NOTE(woosuk): We don't initialize pynccl process group when world size
        # is 1.
        # NOTE(kaichao): By default, pynccl is initialized for tp group.
        pynccl_utils.init_process_group(
            group=get_tensor_model_parallel_cpu_group())

    # Initialize a custom fast all-reduce implementation.
    if not parallel_config.disable_custom_all_reduce:
        init_custom_ar()

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    if pynccl_utils.is_initialized():
        pynccl_utils.all_reduce(torch.zeros(1).cuda())


InterceptObject.add_intercept(worker,
                                  "init_worker_distributed_environment",
                                  sinfer__worker__worker__init_worker_distributed_environment_wraper)
InterceptObject.add_intercept(Worker,
                                  Worker.init_device,
                                  sinfer__worker__worker__Worker__init_device_wraper)
