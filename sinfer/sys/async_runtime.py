import asyncio
import time
from functools import partial
from typing import (Any, AsyncIterator, Callable, Dict, Iterable, List,
                    Optional, Set, Tuple, Type, Union)
from transformers import PreTrainedTokenizer

import sinfer.envs as envs
from sinfer.config import DecodingConfig, ModelConfig
from sinfer.core.scheduler import SchedulerOutputs
from sinfer.sys.arg_utils import AsyncRuntimeArgs
from sinfer.sys.runtime import LLMRuntime
from sinfer.executor.ray_utils import initialize_ray_cluster, ray
from sinfer.logger import init_logger
from sinfer.lora.request import LoRARequest
from sinfer.outputs import RequestOutput
from sinfer.sampling_params import SamplingParams
from sinfer.request import ExecuteModelRequest, SamplerOutput
from sinfer.task import MultiModalData
from sinfer.usage.usage_lib import UsageContext

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.SINFER_ENGINE_ITERATION_TIMEOUT_S


class AsyncRuntimeDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")

    exception = None
    try:
        task.result()
        raise AsyncRuntimeDeadError(msg)
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncRuntimeDeadError(
            msg + " See stack trace above for the actual cause.") from e


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestPool:
    """
    负责管理和追踪多个异步请求的状态和数据流
    """

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(request_id)
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(rid)

    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info("Finished request %s.", request_id)
            self.abort_request(request_id)

    def process_exception(self,
                          request_id: str,
                          exception: Exception,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info("Finished request %s.", request_id)
        self.abort_request(request_id)

    def add_task(self, task_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        if task_id in self._request_streams:
            raise KeyError(f"task {task_id} already exists.")

        stream = AsyncStream(task_id)
        self._new_requests.put_nowait((stream, {
            "task_id": task_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        if verbose:
            logger.info("Aborted request %s.", request_id)

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


class _AsyncLLMRuntime(LLMRuntime):
    """对原始的 LLMRuntime 类进行扩展，增加了异步方法，核心目的是实现异步推理支持"""

    async def step_async(self) -> List[RequestOutput]:
        """异步执行一次解码迭代并返回新生成的结果。
        该函数执行引擎的一次解码迭代。首先，它会调度下一次迭代中需要执行的序列以及需要交换（载入/移出/复制）的 token 块。然后，执行模型推理，并使用模型的输出更新调度器。最后，它对序列进行解码并返回新生成的结果
        """
        task_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            # Execute the model.
            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=task_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
            )
            output = await self.model_executor.execute_model_async(
                execute_model_req)
        else:
            output = []

        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_tasks,
            scheduler_outputs.ignored_tasks, task_metadata_list)

        self.do_log_stats(scheduler_outputs, output)

        return request_outputs

    async def encode_request_async(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = await self.tokenizer.encode_async(
                request_id=request_id,
                prompt=prompt,
                lora_request=lora_request)
        return prompt_token_ids

    async def add_task_async(
        self,
        task_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = await self.encode_request_async(
            request_id=task_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        return self.add_task(task_id,
                                prompt=prompt,
                                prompt_token_ids=prompt_token_ids,
                                sampling_params=sampling_params,
                                arrival_time=arrival_time,
                                lora_request=lora_request,
                                multi_modal_data=multi_modal_data)

    async def check_health_async(self) -> None:
        self.model_executor.check_health()


class AsyncLLMRuntime:
    """对Runtime的异步包装

    这个类用于封装 LLMRuntime 类，使其支持异步操作。  
    它使用 asyncio 创建一个后台循环，用于持续处理传入的请求。  
    当等待队列中有请求时，通过调用 generate 方法唤醒 LLMRuntime 
    generate 方法会将 LLMRuntime 的输出逐步传递给调用者。

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        max_log_len: Maximum number of prompt characters or prompt ID numbers
            being printed in log.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for LLMEngine.
        **kwargs: Arguments for LLMEngine.
    """

    _runtime_class: Type[_AsyncLLMRuntime] = _AsyncLLMRuntime

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_runtime_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.runtime = self._init_runtime(*args, **kwargs)

        self.background_loop: Optional[asyncio.Future] = None

        # 需要保持对未加保护的（unshielded）任务的引用
        # 以防止它被垃圾回收
        self._background_loop_unshielded: Optional[asyncio.Task[Any]] = None
        self.start_runtime_loop = start_runtime_loop
        self._errored_with: Optional[BaseException] = None

        # Lazy initialized fields
        self._request_pool: RequestPool

    @classmethod
    def from_runtime_args(
        cls,
        runtime_args: AsyncRuntimeArgs,
        start_runtime_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLMRuntime":
        """创建一个异步的runtime"""
        runtime_config = runtime_args.create_runtime_config() # 创建一个RuntimeConfig实例

        if runtime_config.device_config.device_type == "cpu":
            assert not runtime_config.parallel_config.worker_use_ray, (
                "Ray is not supported with the CPU backend.")
            from sinfer.executor.cpu_executor import CPUExecutorAsync
            executor_class = CPUExecutorAsync
        elif runtime_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(runtime_config.parallel_config)
            from sinfer.executor.ray_gpu_executor import RayGPUExecutorAsync
            executor_class = RayGPUExecutorAsync
        else:
            assert runtime_config.parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1.")
            from sinfer.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync    # tensor-parallel-size = 1 时
        # 创建异步LLM runtime
        runtime = cls(
            runtime_config.parallel_config.worker_use_ray,
            runtime_args.engine_use_ray,
            **runtime_config.to_dict(),
            executor_class=executor_class,
            log_requests=not runtime_args.disable_log_requests,
            log_stats=not runtime_args.disable_log_stats,
            max_log_len=runtime_args.max_log_len,
            start_runtime_loop=start_runtime_loop,
            usage_context=usage_context,
        )
        return runtime

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())

    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None and
                                self._background_loop_unshielded is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_pool.propagate_exception(exc)

    async def get_tokenizer(self) -> "PreTrainedTokenizer":
        if self.engine_use_ray:
            return await self.runtime.get_tokenizer.remote()  # type: ignore
        else:
            return self.runtime.get_tokenizer()

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            logger.info(self._errored_with)
            raise AsyncRuntimeDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_pool = RequestPool()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.runtime_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_runtime(self, *args,
                     **kwargs) -> Union[_AsyncLLMRuntime, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            runtime_class = self._runtime_class
        elif self.worker_use_ray:
            runtime_class = ray.remote(num_cpus=0)(self._runtime_class).remote
        else:
            # This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = kwargs["cache_config"]
            parallel_config = kwargs["parallel_config"]
            if parallel_config.tensor_parallel_size == 1:
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            runtime_class = ray.remote(num_gpus=num_gpus)(
                self._runtime_class).remote
        return runtime_class(*args, **kwargs)

    async def runtime_step(self) -> bool:
        """触发引擎处理tracker中的请求
        如果有正在处理的请求，则返回 True"""

        new_requests, finished_requests = (
            self._request_pool.get_new_and_finished_requests())

        for new_request in new_requests:
            # 将请求添加到 SInfer 引擎的等待队列中
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            try:
                if self.engine_use_ray:
                    await self.runtime.add_task.remote(  # type: ignore
                        **new_request)
                else:
                    await self.runtime.add_task_async(**new_request)
            except ValueError as e:
                # TODO: use a SInfer specific error for failed validation
                self._request_pool.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        if finished_requests:
            await self._runtime_abort(finished_requests)

        if self.engine_use_ray:
            request_outputs = await self.runtime.step.remote()  # type: ignore
        else:
            request_outputs = await self.runtime.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_pool.process_request_output(
                request_output, verbose=self.log_requests)

        return len(request_outputs) > 0

    async def _runtime_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.runtime.abort_task.remote(request_ids)  # type: ignore
        else:
            self.runtime.abort_task(request_ids)

    async def runtime_loop(self):
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                logger.debug("Waiting for new requests...")
                await self._request_pool.wait_for_new_requests()
                logger.debug("Got new requests!")

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                has_requests_in_progress = await asyncio.wait_for(
                    self.runtime_step(), ENGINE_ITERATION_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Engine iteration timed out. This should never happen!")
                self.set_errored(exc)
                raise
            await asyncio.sleep(0)

    async def add_task(
        self,
        task_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(
                "Received request %s: prompt: %r, "
                "sampling_params: %s, prompt_token_ids: %s, "
                "lora_request: %s.", task_id, shortened_prompt,
                sampling_params, shortened_token_ids, lora_request)

        if not self.is_running:
            if self.start_runtime_loop:
                self.start_background_loop()
            else:
                raise AsyncRuntimeDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        if arrival_time is None:
            arrival_time = time.time()

        if self.engine_use_ray:
            prompt_token_ids = await (
                self.runtime.encode_request_async.remote(  # type: ignore
                    request_id=task_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    lora_request=lora_request))
        else:
            prompt_token_ids = await self.runtime.encode_request_async(
                request_id=task_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                lora_request=lora_request)

        stream = self._request_pool.add_task(
            task_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
        )

        return stream

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            lora_request: LoRA request to use for generation, if any.
            multi_modal_data: Multi modal data per request.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~sinfer.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        # Preprocess the request.
        arrival_time = time.time()

        try:
            stream = await self.add_task(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
                lora_request=lora_request,
                multi_modal_data=multi_modal_data,
            )

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncRuntimeDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """终止一个已提交的request. 
        If the request is finished or not found,
        this method will be a no-op.

        """
        self._request_pool.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the SInfer engine."""
        if self.engine_use_ray:
            return await self.runtime.get_model_config.remote()  # type: ignore
        else:
            return self.runtime.get_model_config()

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the SInfer engine."""
        if self.engine_use_ray:
            return await self.runtime.get_decoding_config.remote(  # type: ignore
            )
        else:
            return self.runtime.get_decoding_config()

    async def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        if self.engine_use_ray:
            await self.runtime.do_log_stats.remote(  # type: ignore
                scheduler_outputs, model_output)
        else:
            self.runtime.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if runtime is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncRuntimeDeadError("Background loop is stopped.")

        if self.engine_use_ray:
            try:
                await self.runtime.check_health.remote()  # type: ignore
            except ray.exceptions.RayActorError as e:
                raise RuntimeError("Engine is dead.") from e
        else:
            await self.runtime.check_health_async()
        logger.debug("Health check took %fs", time.perf_counter() - t)

    def save_scheduler_view(self):
        if hasattr(self.runtime.scheduler, "__del__"):
            self.runtime.scheduler.__del__()