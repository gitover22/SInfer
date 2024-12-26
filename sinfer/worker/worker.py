import gc
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed

from sinfer.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from sinfer.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              get_tensor_model_parallel_cpu_group,
                              init_distributed_environment)
from sinfer.distributed.device_communicators import pynccl_utils
from sinfer.distributed.device_communicators.custom_all_reduce import (
    init_custom_ar)
from sinfer.lora.request import LoRARequest
from sinfer.model_executor import set_random_seed
from sinfer.request import ExecuteModelRequest, SamplerOutput
from sinfer.worker.cache_engine import CacheEngine
from sinfer.worker.model_runner import ModelRunner
from sinfer.worker.worker_base import WorkerBase


class Worker(WorkerBase):
    """A worker class that executes (a partition of) the model on a MLU.

    Each worker is associated with a single MLU. The worker is responsible for
    maintaining the KV cache and executing the model on the MLU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from sinfer.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        self.model_runner = ModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config,
        )

        self.cache_engine: CacheEngine  # 没初始化
        self.gpu_cache: List[torch.Tensor]

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce 在同步点之前不会释放输入张量
            # 这会导致随着 all_reduce 调用次数的增加，内存使用量不断增长。
            # 通过设置该环境变量可以禁用这种行为
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None) # 删掉一个Ray产生的环境变量
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # 初始化分布式环境--》 模型并行组划分等（动态划分，根据设备数量）
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # 设置随机种子
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
            分析模型的峰值内存使用情况, 以确定可以分配多少个 KV 块而不会发生内存溢出(OOM)
            引擎会首先对现有的内存使用情况进行分析,然后计算在剩余的可用内存中设备可分配的最大块数量。
        """
        # 分析模型的内存使用情况，获取在剩余空闲内存中可以分配的缓存块的最大数量
        torch.cuda.empty_cache() # 清理 CUDA 缓存内存
        # 执行一次前向传递，使用虚拟输入以分析模型的内存使用情况。
        self.model_runner.profile_run()
        torch.cuda.synchronize() # 同步 CUDA 操作，确保内存分析准确
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info() # 24.9GB  41.5GB
        # 假设在内存分析期间使用同一 GPU 的其他进程没有改变其内存使用情况
        peak_memory = self.init_gpu_memory - free_gpu_memory  # 16.5GB
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the SInfer instance.")

        cache_block_size = self.get_cache_block_size_bytes() # 获取单个缓存块的大小 128MB
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def _warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # 重置随机数种子，以确保随机状态不受模型初始化和分析的影响。
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # 
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:

        if execute_model_req is None:
            seq_group_metadata_list = None
        else:
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            assert execute_model_req is not None
            num_seq_groups = len(seq_group_metadata_list)
            blocks_to_swap_in = execute_model_req.blocks_to_swap_in
            blocks_to_swap_out = execute_model_req.blocks_to_swap_out
            blocks_to_copy = execute_model_req.blocks_to_copy
            data: Dict[str, Any] = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0) # 广播给其他 Worker
        else:
            data = broadcast_tensor_dict(src=0) # 接收 Driver 广播的数据
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # 没有输入数据，直接返回空列表
        if num_seq_groups == 0:
            return []
        # 调用ModelRunner执行模型
        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)

        # Worker 仅支持单步执行，将输出包装为一个列表以符合接口要求
        return [output]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """获取kv cache的块大小bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)

def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)
    # 确保模型的并行化已初始化
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    if pynccl_utils.is_initialized():
        pynccl_world_size = pynccl_utils.get_world_size()
        if pynccl_world_size != parallel_config.world_size:
            raise RuntimeError(
                "pynccl is already initialized but the pynccl world "
                "size does not match parallel_config.world_size "
                f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    elif parallel_config.world_size > 1:
        # 当 world size 为 1 时，不会初始化 pynccl 进程组.
        # 默认情况下，pynccl 为 tp group 初始化
        pynccl_utils.init_process_group(
            group=get_tensor_model_parallel_cpu_group())

    # 初始化自定义的all-reduce
    if not parallel_config.disable_custom_all_reduce:
        init_custom_ar()

    # 执行 all-reduce，用于预热分布式训练中的通信系统
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    if pynccl_utils.is_initialized():
        pynccl_utils.all_reduce(torch.zeros(1).cuda())


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8: # 算力在8.0以上才支持BF16
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
