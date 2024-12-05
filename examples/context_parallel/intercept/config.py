from sinfer.config import ParallelConfig, TokenizerPoolConfig
from typing import Optional
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject

def sinfer__config__ParallelConfig___init__(
    self,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
    worker_use_ray: bool,
    max_parallel_loading_workers: Optional[int] = None,
    disable_custom_all_reduce: bool = False,
    tokenizer_pool_config: Optional[TokenizerPoolConfig] = None,
    ray_workers_use_nsight: bool = False,
    placement_group: Optional["PlacementGroup"] = None,
) -> None:
    self.pipeline_parallel_size = pipeline_parallel_size
    self.tensor_parallel_size = tensor_parallel_size
    self.context_parallel_size = self.context_parallel_size
    self.worker_use_ray = worker_use_ray
    self.max_parallel_loading_workers = max_parallel_loading_workers
    self.disable_custom_all_reduce = disable_custom_all_reduce
    self.tokenizer_pool_config = tokenizer_pool_config
    self.ray_workers_use_nsight = ray_workers_use_nsight
    self.placement_group = placement_group

    '''
    ==========================
    Modify by Context Parallel
    ==========================
    @brief: context parallel world_size
    '''
    self.world_size = pipeline_parallel_size * tensor_parallel_size * self.context_parallel_size
    '''
    =======================
    End of Context Parallel
    =======================
    '''
    
    if self.world_size > 1:
        self.worker_use_ray = True
    self._verify_args()
    

MluHijackObject.add_hijack_object(ParallelConfig,
                                  ParallelConfig.__init__,
                                  sinfer__config__ParallelConfig___init__)