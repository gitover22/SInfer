from typing import Any, Dict, List, Optional, Set, Tuple

from sinfer.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from sinfer.logger import init_logger
from sinfer.lora.request import LoRARequest
from sinfer.sequence import ExecuteModelRequest, SamplerOutput
from sinfer.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from sinfer.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class GPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        """
        初始化worker并加载模型
        """
        self._init_worker()

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=rank == 0,
        )

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):
        wrapper = WorkerWrapperBase(
            worker_module_name="sinfer.worker.worker",
            worker_class_name="Worker",
        )
        wrapper.init_worker(**self._get_worker_kwargs(local_rank, rank,
                                                      distributed_init_method))
        return wrapper.worker

    def _init_worker(self):
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = self._create_worker() # 创建worker
        self.driver_worker.init_device() # 初始化设备
        self.driver_worker.load_model() # 加载模型

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """调用Worker来确定blocks数量
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        logger.info("Allocated GPU blocks: %d, CPU blocks: %d", num_gpu_blocks, num_cpu_blocks)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output
