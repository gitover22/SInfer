from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

from sinfer.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from sinfer.lora.request import LoRARequest
from sinfer.request import ExecuteModelRequest, SamplerOutput


class InterfaceBase(ABC):
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig]
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.vision_language_config = vision_language_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        pass

    @abstractmethod
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        raise NotImplementedError

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        return

    def __del__(self):
        self.shutdown()

class InterfaceAsyncBase(InterfaceBase):
    @abstractmethod
    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        raise NotImplementedError

    async def check_health_async(self) -> None:
        self.check_health()
