import time
from typing import Iterable, List, Optional, Type, Union

from transformers import GenerationConfig, PreTrainedTokenizer

import sinfer
from sinfer.task import MultiModalData, Task, TaskMetadata
from sinfer.config import (CacheConfig, DecodingConfig, DeviceConfig, LoadConfig,
                         LoRAConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VisionLanguageConfig)
from sinfer.scheduler.TAScheduler import (ScheduledTask, Scheduler,
                                 SchedulerOutputs)
from sinfer.sys.runtime_args import RuntimeArgs
from sinfer.sys.metrics import StatLogger, Stats
from sinfer.sys.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from sinfer.sys.output_processor.stop_checker import StopChecker
from sinfer.sys.output_processor.util import create_output_by_task
from sinfer.hiLayer.InterfaceBase import InterfaceBase
from sinfer.hiLayer.init_ray import initialize_ray_cluster
from sinfer.logger import init_logger
from sinfer.lora.request import LoRARequest
from sinfer.outputs import RequestOutput
from sinfer.sampling_params import SamplingParams
from sinfer.request import (ExecuteModelRequest, SamplerOutput,
                           Request, RequestStatus)
from sinfer.transformers_utils.detokenizer import Detokenizer
from sinfer.transformers_utils.tokenizer_group import (BaseTokenizerGroup,
                                                     get_tokenizer_group)
from sinfer.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from sinfer.utils import Counter

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5

def _load_generation_config_dict(model_config: ModelConfig):
    try:
        return GenerationConfig.from_pretrained(
            model_config.model,
            revision=model_config.revision,
        ).to_diff_dict()
    except OSError:
        # Not found.
        return {}


class LLMRuntime:
    """
    参数:
        model_config: LLM model 配置.
        cache_config: KV cache内存管理配置.
        parallel_config: distributed execution配置.
        scheduler_config: request scheduler配置.
        device_config: device配置.
        lora_config (Optional): serving multi-LoRA配置.
        vision_language_config (Optional): vision language models配置
        speculative_config (Optional): speculative decoding配置.
        executor_class: The model executor class for managing distributed execution.
        log_stats: Whether to log statistics.
        usage_context: Specified entry point, used for usage info collection
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        decoding_config: Optional[DecodingConfig],
        interfacer: Type[InterfaceBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> None:
        logger.info(
            "Initializing an LLM runtime (v%s) with config: model=%r, tokenizer=%r,"
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "tokenizer_revision=%s, trust_remote_code=%s, dtype=%s, "
            "max_seq_len=%d, download_dir=%r, load_format=%s, "
            "tensor_parallel_size=%d, disable_custom_all_reduce=%s, "
            "quantization=%s, enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, seed=%d, served_model_name=%s)",
            sinfer.__version__,
            model_config.model,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            model_config.tokenizer_mode,
            model_config.revision,
            model_config.tokenizer_revision,
            model_config.trust_remote_code,
            model_config.dtype,
            model_config.max_model_len,
            load_config.download_dir,
            load_config.load_format,
            parallel_config.tensor_parallel_size,
            parallel_config.disable_custom_all_reduce,
            model_config.quantization,
            model_config.enforce_eager,
            cache_config.cache_dtype,
            model_config.quantization_param_path,
            device_config.device,
            decoding_config,
            model_config.seed,
            model_config.served_model_name,
        )
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.vision_language_config = vision_language_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.load_config = load_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.log_stats = log_stats

        if not self.model_config.skip_tokenizer_init:
            self.tokenizer: BaseTokenizerGroup
            self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)
        else:
            self.detokenizer = None
            self.tokenizer = None

        self.req_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(
            model_config)

        self._interfacer = interfacer(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            vision_language_config=vision_language_config,
            load_config=load_config,
        )

        self._preallocate_kv_caches() # 初始化kv cache缓存，CacheManager

        # 收集信息
        if is_usage_stats_enabled():
            from sinfer.model_executor.model_loader import (
                get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype":
                    str(model_config.dtype),
                    "tensor_parallel_size":
                    parallel_config.tensor_parallel_size,
                    "block_size":
                    cache_config.block_size,
                    "mlu_memory_utilization":
                    cache_config.mlu_memory_utilization,

                    # Quantization
                    "quantization":
                    model_config.quantization,
                    "kv_cache_dtype":
                    cache_config.cache_dtype,

                    # Feature flags
                    "enable_lora":
                    bool(lora_config),
                    "enable_prefix_caching":
                    cache_config.enable_prefix_caching,
                    "enforce_eager":
                    model_config.enforce_eager,
                    "disable_custom_all_reduce":
                    parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a different process.
            self.tokenizer.ping()

        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        # Metric Logging.
        if self.log_stats:
            self.stat_logger = StatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                labels=dict(model_name=model_config.served_model_name),
                max_model_len=self.model_config.max_model_len)
            self.stat_logger.info("cache_config", self.cache_config)

        # 创建序列输出处理器
        self.output_processor = (
            SequenceGroupOutputProcessor.create_output_processor(
                self.scheduler_config,
                self.detokenizer,
                self.scheduler,
                self.req_counter,
                self.get_tokenizer_for_seq,
                stop_checker=StopChecker(
                    self.scheduler_config.max_model_len,
                    self.get_tokenizer_for_seq,
                ),
            ))

    def _preallocate_kv_caches(self) -> None:
        num_mlu_blocks, num_cpu_blocks = (
            self._interfacer.determine_num_available_blocks())

        if self.cache_config.num_mlu_blocks_override is not None:
            num_mlu_blocks_override = self.cache_config.num_mlu_blocks_override
            logger.info(
                "Overriding num_mlu_blocks=%d with "
                "num_mlu_blocks_override=%d", num_mlu_blocks,
                num_mlu_blocks_override)
            num_mlu_blocks = num_mlu_blocks_override

        self.cache_config.num_mlu_blocks = num_mlu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._interfacer.initialize_cache(num_mlu_blocks, num_cpu_blocks)

    @classmethod
    def from_runtime_args(
        cls,
        runtime_args: RuntimeArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "LLMRuntime":
        runtime_config = runtime_args.create_runtime_config()

        if runtime_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(runtime_config.parallel_config)
            from sinfer.hiLayer.ray_MLUInterface import Ray_MLUInterface
            interfacer = Ray_MLUInterface
        else:
            assert runtime_config.parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1.")
            from sinfer.hiLayer.mlu_Interface import MLUInterface
            interfacer = MLUInterface
        # if runtime_config.device_config.device_type == "cpu":
        #     from sinfer.hil.cpu_Interface import CPUInterface
        #     interfacer = CPUInterface
        # elif runtime_config.parallel_config.worker_use_ray:
        #     initialize_ray_cluster(runtime_config.parallel_config)
        #     from sinfer.hil.ray_MLUInterface import Ray_MLUInterface
        #     interfacer = Ray_MLUInterface
        # else:
        #     assert runtime_config.parallel_config.world_size == 1, (
        #         "Ray is required if parallel_config.world_size > 1.")
        #     from sinfer.hil.mlu_Interface import MLUInterface
        #     interfacer = MLUInterface

        # 运行构造函数创建runtime对象
        runtime = cls(
            **runtime_config.to_dict(),
            interfacer=interfacer,
            log_stats=not runtime_args.disable_log_stats,
            usage_context=usage_context,
        )
        return runtime

    def __reduce__(self):
        # 确保 LLMRuntime 对象不会被序列化
        raise RuntimeError("LLMRuntime should not be pickled!")

    def __del__(self):
        # 当引擎被垃圾回收时关闭model executor
        if model_executor := getattr(self, "model_executor", None):
            model_executor.shutdown()

    def get_tokenizer(self) -> "PreTrainedTokenizer":
        return self.tokenizer.get_lora_tokenizer(None)

    def get_tokenizer_for_seq(self,
                              sequence: Request) -> "PreTrainedTokenizer":
        return self.tokenizer.get_lora_tokenizer(sequence.lora_request)

    def _init_tokenizer(self, **tokenizer_init_kwargs):
        init_kwargs = dict(
            tokenizer_id=self.model_config.tokenizer,
            enable_lora=bool(self.lora_config),
            max_num_seqs=self.scheduler_config.max_num_reqs,
            max_input_length=None,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision)
        init_kwargs.update(tokenizer_init_kwargs)
        self.tokenizer = get_tokenizer_group(
            self.parallel_config.tokenizer_pool_config, **init_kwargs)

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config:
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)

    def encode_request(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(request_id=request_id,
                                                     prompt=prompt,
                                                     lora_request=lora_request)
        return prompt_token_ids

    def add_task(
        self,
        task_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        """
        将一个request添加给LLMRuntime
        Args:
            task_id: task的id
            prompt: prompt( str )
            sampling_params: 采样参数(温度、topk等)
            prompt_token_ids: prompt(token_id版), 没有提供的话会调用tokenizer来做
            arrival_time: 请求到达的时间。如果是None,则用当前系统时间
            lora_request: 如果是用lora模型做推理,相关的lora请求
            multi_modal_data: 多模态数据
        """
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        max_logprobs = self.get_model_config().max_logprobs
        if (sampling_params.logprobs
                and sampling_params.logprobs > max_logprobs) or (
                    sampling_params.prompt_logprobs
                    and sampling_params.prompt_logprobs > max_logprobs):
            raise ValueError(f"Cannot request more than "
                             f"{max_logprobs} logprobs.")
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = self.encode_request(
            request_id=task_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        # 将用户prompt包装为Request
        block_size = self.cache_config.block_size
        req_id = next(self.req_counter) # 获取Request对应的id
        eos_token_id = None
        if self.tokenizer:
            eos_token_id = self.tokenizer.get_lora_tokenizer(
                lora_request).eos_token_id
        else:
            logger.warning("Use None for EOS token id because tokenizer is "
                           "not initialized")
        req = Request(req_id, prompt, prompt_token_ids, block_size,
                       eos_token_id, lora_request)

        # 对SamplingParams进行防御性拷贝，这些参数由采样器使用，但不会对LogitsProcessor对象进行深拷贝
        sampling_params = sampling_params.clone()
        
        if req.eos_token_id is not None:
            sampling_params.all_stop_token_ids.add(req.eos_token_id)  # 将eos token ID添加到sampling_params中，以支持min_tokens的处理
        sampling_params.update_from_generation_config(
            self.generation_config_fields)

        # 检查 input_len + output_len <= block_size(非分页模式下，缓存的块大小需要能够容纳输入和输出的token)
        num_prompt_tokens = 0
        out_len = sampling_params.max_tokens # 生成的最大token数量

        from sinfer.Intercept.mlu_definitions import BlockSizeInfo, USE_PAGED
        if (USE_PAGED == False and BlockSizeInfo.BLOCK_SIZE < out_len + req.get_len()):
            raise ValueError("BLOCK_SIZE(" + str(BlockSizeInfo.BLOCK_SIZE) +
                ") can't smaller than input_len(" + str(req.get_len()) + ") + output_len(" + str(out_len) + ").")

        # 使用req包装为Task,每个prompt被包装成一个Task实例
        task = Task(task_id, [req], sampling_params,
                                  arrival_time, lora_request, multi_modal_data)

        # 向调度器添加序列组, 添加进scheduler的self.waiting队列中
        self.scheduler.add_task(task)

    def abort_task(self, task_id: Union[str, Iterable[str]]) -> None:
        """中止给定 ID 的请求
        Args:
            task_id: The ID(s) of the task to abort.
        """
        self.scheduler.abort_task(task_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_decoding_config(self) -> DecodingConfig:
        """Gets the decoding configuration."""
        return self.decoding_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_tasks()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_tasks()

    def _process_model_outputs(
        self,
        output: List[SamplerOutput],
        scheduled_tasks: List[ScheduledTask],
        ignored_tasks: List[Task],
        task_metadata_list: List[TaskMetadata],
    ) -> List[RequestOutput]:
        """Apply the model output to the requests in the scheduled tasks.
        Returns RequestOutputs that can be returned to the client.
        """

        now = time.time()

        # Organize outputs by [task][step] instead of [step][task].
        output_by_task = create_output_by_task(
            sampler_outputs=output, num_tasks=len(scheduled_tasks))

        # Update the scheduled tasks with the model outputs.
        for scheduled_task, outputs, task_meta in zip(
                scheduled_tasks, output_by_task,task_metadata_list):
            task = scheduled_task.task
            task.update_num_computed_tokens(
                scheduled_task.token_chunk_size)

            self.output_processor.process_prompt_logprob(task, outputs)
            if task_meta.do_sample:
                self.output_processor.process_outputs(task, outputs)

        # Free finished tasks
        self.scheduler.free_all_finished_tasks()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for scheduled_task in scheduled_tasks:
            task = scheduled_task.task
            task.maybe_set_first_token_time(now)
            request_output = RequestOutput.from_task(task)
            request_outputs.append(request_output)
        for task in ignored_tasks:
            request_output = RequestOutput.from_task(task)
            request_outputs.append(request_output)
        return request_outputs

    def step(self) -> List[RequestOutput]:
        task_metadata_list, scheduler_outputs = self.scheduler.schedule_entrance()

        if not scheduler_outputs.is_empty():
            execute_model_req = ExecuteModelRequest(
                task_metadata_list=task_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
            )
            output = self._interfacer.execute_model(
                execute_model_req=execute_model_req)
        else:
            output = []

        request_outputs = self._process_model_outputs(
                            output, 
                            scheduler_outputs.scheduled_tasks,
                            scheduler_outputs.ignored_tasks, 
                            task_metadata_list)

        
        self.do_log_stats(scheduler_outputs, output)

        return request_outputs

    def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        # 当没有请求时强制日志
        if self.log_stats:
            self.stat_logger.log(
                self._get_stats(scheduler_outputs, model_output))

    def _get_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs],
            model_output: Optional[List[SamplerOutput]] = None) -> Stats:
        now = time.time()
        num_running_sys = len(self.scheduler.RunningPool)
        num_swapped_sys = len(self.scheduler.BlockedPool)
        num_waiting_sys = len(self.scheduler.ReadyPool)

        # KV Cache Usage
        _total_mlu_num = self.cache_config.num_mlu_blocks
        _free_mlu_num = self.scheduler.block_manager.get_num_free_mlu_blocks()
        mlu_cache_usage_sys = 1.0 - (_free_mlu_num / _total_mlu_num)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage_sys = 0.
        if num_total_cpu > 0:
            num_free_cpu = self.scheduler.block_manager.get_num_free_cpu_blocks()
            
            cpu_cache_usage_sys = 1.0 - (num_free_cpu / num_total_cpu)

        # Iteration stats
        num_prompt_tokens_iter = 0
        num_generation_tokens_iter = 0
        time_to_first_tokens_iter: List[float] = []   # ttft
        time_per_output_tokens_iter: List[float] = [] # tpot

       
        
        time_e2e_requests: List[float] = [] # Latency:e2e
        
        _prompt_tokens_num: List[int] = []
        _generation_tokens_num: List[int] = []
        best_of_requests: List[int] = []
        n_requests: List[int] = []
        finished_reason_requests: List[str] = []

        # This loop assumes prefill tasks are before
        # decode tasks in scheduled_tasks.
        if scheduler_outputs is not None:
            num_generation_tokens_from_prefill_groups = 0.
            # NOTE: if scheduler_outputs.num_prefill_groups > 0 and
            # the len of scheduler_outputs.scheduled_tasks is !=
            # scheduler_outputs.num_prefill_groups, this means that
            # chunked prefills have been detected.

            for idx, scheduled_task in enumerate(
                    scheduler_outputs.scheduled_tasks):
                group_was_prefill = idx < scheduler_outputs.num_prefill_tasks
                task = scheduled_task.task

                # NOTE: a task that completed all of its prefill tokens
                # in the last iteration will have task.is_prefill() = False
                # with group_was_prefill = True
                if group_was_prefill:
                    # Number of prompt tokens.
                    num_prompt_tokens_iter += (
                        scheduled_task.token_chunk_size)

                    # If the task just finished the prefill state
                    # get TTFT.
                    if not task.is_prefill():
                        latency = task.get_last_latency(now)
                        time_to_first_tokens_iter.append(latency)

                        # One generation token per finished prefill.
                        num_generation_tokens_from_prefill_groups += (
                            task.num_reqs())
                else:
                    # TPOTs.
                    latency = task.get_last_latency(now)
                    time_per_output_tokens_iter.append(latency)

                # Because of chunked prefill, we can have a single task
                # that does multiple prompt_runs. To prevent logging
                # the same metadata more than once per request, we standardize
                # on logging request level information for finished requests,
                # which can only happen once.
                if task.is_finished():
                    # Latency timings
                    time_e2e_requests.append(now - task.metadata.arrival_time)

                    # Metadata
                    _prompt_tokens_num.append(
                        len(task.prompt_token_ids))
                    _generation_tokens_num.extend([
                        seq.get_output_len()
                        for seq in task.get_finished_reqs()
                    ])
                    best_of_requests.append(task.sampling_params.best_of)
                    n_requests.append(task.sampling_params.n)
                    finished_reason_requests.extend([
                        RequestStatus.get_finished_reason(seq.status)
                        for seq in task.get_finished_reqs()
                    ])

            # Number of generation tokens.
            #   num_batched_tokens equals the number of prompt_tokens plus the
            #   number of decode_tokens in a single iteration. So,
            #   num_generation_tokens = num_batched_tokens - num_prompt_tokens
            #   + num_generation_tokens_from_prefill_groups (since we generate
            #   one token on prefills on iters where the prefill finishes).
            num_generation_tokens_iter = (
                scheduler_outputs.num_batched_tokens - num_prompt_tokens_iter +
                num_generation_tokens_from_prefill_groups)


        return Stats(
            now=now,

            # System stats
            #   Scheduler State
            num_running_sys=num_running_sys,
            num_blocked_sys=num_swapped_sys,
            num_ready_sys=num_waiting_sys,
            #   KV Cache Usage in %
            mlu_cache_usage_sys=mlu_cache_usage_sys,
            cpu_cache_usage_sys=cpu_cache_usage_sys,

            # Iteration stats
            num_prompt_tokens_iter=num_prompt_tokens_iter,
            num_generation_tokens_iter=num_generation_tokens_iter,
            time_to_first_tokens_iter=time_to_first_tokens_iter,
            time_per_output_tokens_iter=time_per_output_tokens_iter,

            # Request stats
            #   Latency
            time_e2e_requests=time_e2e_requests,
            #   Metadata
            num_prompt_tokens_requests=_prompt_tokens_num,
            num_generation_tokens_requests=_generation_tokens_num,
            best_of_requests=best_of_requests,
            n_requests=n_requests,
            finished_reason_requests=finished_reason_requests,
        )

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self._interfacer.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self._interfacer.remove_lora(lora_id)

    def list_loras(self) -> List[int]:
        return self._interfacer.list_loras()

    def check_health(self) -> None:
        self._interfacer.check_health()

    def intercept_get_latency(self):
        latency = self._interfacer.get_latency()
        return latency
