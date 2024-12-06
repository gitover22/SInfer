import torch
import torch.distributed
from sinfer.distributed import init_distributed_environment
from sinfer.worker.cpu_worker import CPUWorker
from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from ..distributed.parallel_state import sinfer__distributed__parallel_state__ensure_model_parallel_initialized_wraper


def sinfer__worker__worker__CPUWorker__init_distributed_environment_wraper(self) -> None:
    """Initialize the distributed environment."""

    parallel_config = self.parallel_config
    rank = self.rank
    distributed_init_method = self.distributed_init_method
    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        backend="gloo",
    )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cpu())

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


InterceptObject.add_intercept(CPUWorker,
                                  CPUWorker.init_distributed_environment,
                                  sinfer__worker__worker__CPUWorker__init_distributed_environment_wraper)
