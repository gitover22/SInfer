from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from sinfer.hil.InterfaceBase import InterfaceAsyncBase
from sinfer.hil.mlu_Interface import MLUInterface
from sinfer.logger import init_logger
from sinfer.lora.request import LoRARequest
from sinfer.request import SamplerOutput

logger = init_logger(__name__)

class Distributed_MLUInterface(MLUInterface):
    """多 MLU 执行器实现的抽象父类"""
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """ 确定可用的 KV 缓存块数量

        此方法会在每个节点上调用 `determine_num_available_blocks`，并取结果的最小值，
        从而保证所选的缓存大小对所有节点来说都是兼容的。

        Returns:
            - tuple[num_mlu_blocks, num_cpu_blocks] MLU 和 CPU 上的可用块数量
        """
        # 获取 mlu 和 CPU 上可以分配的最大块数量
        num_blocks = self._run_workers("determine_num_available_blocks", )

        # 由于使用的是共享的集中控制器，这里取所有节点的块数量的最小值，
        # 确保所有内存算子可以应用于所有节点
        num_mlu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_mlu_blocks, num_cpu_blocks

    def initialize_cache(self, num_mlu_blocks: int,
                         num_cpu_blocks: int) -> None:
        logger.info("# MLU kv blocks: %d, # CPU kv blocks: %d", num_mlu_blocks,
                    num_cpu_blocks)

        self.cache_config.num_mlu_blocks = num_mlu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._run_workers("initialize_cache",
                          num_mlu_blocks=num_mlu_blocks,
                          num_cpu_blocks=num_cpu_blocks)

    def execute_model(self, *args, **kwargs) -> List[SamplerOutput]:
        all_outputs = self._run_workers("execute_model",
                                        driver_args=args,
                                        driver_kwargs=kwargs)
        # 只有主节点返回
        return all_outputs[0]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "add_lora",
            lora_request=lora_request,
        )

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "remove_lora",
            lora_id=lora_id,
        )

    def list_loras(self) -> Set[int]:
        return self._run_workers("list_loras")

    @abstractmethod
    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        # 在所有节点上运行method
        raise NotImplementedError

class Distributed_MLUInterfaceAsync(Distributed_MLUInterface, InterfaceAsyncBase):
    @abstractmethod
    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """让所有节点运行method方法"""
        raise NotImplementedError

    async def execute_model_async(self, *args,
                                  **kwargs) -> List[SamplerOutput]:
        all_outputs = await self._run_workers_async("infer_model",
                                                    driver_args=args,
                                                    driver_kwargs=kwargs)
        # 只有主节点返回结果
        return all_outputs[0]
