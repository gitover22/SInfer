from typing import List, Optional, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sinfer.sys.arg_utils import RuntimeArgs
from sinfer.sys.runtime import LLMRuntime
from sinfer.lora.request import LoRARequest
from sinfer.outputs import RequestOutput
from sinfer.sampling_params import SamplingParams
from sinfer.task import MultiModalData
from sinfer.usage.usage_lib import UsageContext
from sinfer.utils import Counter
from sinfer.logger import init_logger
logger = init_logger(__name__)

class LLM:
    """
    包装LLMRuntime, 用于离线推理的用户接口类

    参数:
        model: 模型名或路径.
        tokenizer: 分词器名称或路径
        tokenizer_mode: 分词器模式,“auto”将会使用快速分词器(如果可用),而“slow”将始终使用慢速分词器。
        skip_tokenizer_init: 如果为 True,将跳过分词器和反分词器的初始化。需要在输入中提供有效的 prompt_token_ids,并将 prompt 设置为 None
        trust_remote_code: 在下载模型和分词器时是否信任远程代码(例如来自 HuggingFace)
        tensor_parallel_size: 使用TP并行的设备数量
        dtype: 模型权重和激活值的数据类型。目前支持 float32、float16 和 bfloat16。如果设置为 auto,则使用模型配置文件中的 torch_dtype 属性。但是,如果配置中的 torch_dtype 是 float32,将改为使用 float16
        quantization: 用于量化模型权重的方法。目前支持 "awq"、"gptq"、"squeezellm"、"weightonly" 和 "fp8"(测试中)。如果为 None,将首先检查模型配置文件中的 quantization_config 属性。如果该属性为 None,则假定模型权重未被量化,并使用 dtype 来确定权重的数据类型。
        revision: 使用的具体模型版本。可以是分支名、标签名或提交 ID
        tokenizer_revision: 使用的具体分词器版本。可以是分支名、标签名或提交 ID
        seed: 用于初始化随机数生成器的采样种子
        gpu_memory_utilization: GPU 内存的利用率(在 0 到 1 之间)。用于保留模型权重、激活值和 KV 缓存的比例。
                                更高的值将增加 KV 缓存的大小,从而提高模型的吞吐量。但是,值过高可能会导致内存不足(OOM)错误。
        swap_space: 每个 GPU 使用的 CPU 内存大小(GB)作为交换空间。用于临时存储请求的状态,当它们的 best_of 参数大于 1 时非常有用。
                    如果所有请求的 best_of=1,可以将其安全设置为 0。否则,值过小可能会导致OOM
        enforce_eager: 是否强制使用即时模式执行。如果设置为 True,将禁用 CUDA 图,并始终在即时模式下执行模型。如果设置为 False,则在 CUDA 图和即时执行之间使用混合模式
        max_context_len_to_capture: CUDA 图覆盖的最大上下文长度。当序列的上下文长度超过该值时,将回退到即时模式(已弃用,建议使用 max_seq_len_to_capture 代替)
        max_seq_len_to_capture: CUDA 图覆盖的最大序列长度。当序列的上下文长度超过该值时,将回退到即时模式
        disable_custom_all_reduce: 是否禁用自定义all reduce操作
    """
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        '''
        支持用户定义最大捕获batch大小, 并初始化LLMDumpInfo
        '''
        from sinfer.Intercept.mlu_metric import LLMMetric
        from sinfer.Intercept.dump_info import LLMDumpInfo
        self.metric = LLMMetric()
        self.dump_info = LLMDumpInfo()
        if "max_batch_size_to_capture" in kwargs:
            max_batch_size_to_capture = kwargs.pop("max_batch_size_to_capture")
            if max_batch_size_to_capture:
                from sinfer.Intercept.intercept_utils import set_batch_sizes_to_capture
                set_batch_sizes_to_capture(max_batch_size_to_capture)
        self.dump_info.init_param(tensor_parallel_size=tensor_parallel_size, dtype=dtype, 
                kv_cache_dtype=kwargs.get('kv_cache_dtype', 'default_value'),
                quantization=quantization,
                model=model, trust_remote_code=kwargs.get('trust_remote_code', 'default_value'))
        
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        runtime_args = RuntimeArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_runtime = LLMRuntime.from_runtime_args(
            runtime_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()
        self.dump_info.memory_usage()

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_runtime.tokenizer.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_runtime.tokenizer.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> List[RequestOutput]:
        """
        Args:
            prompts: 提示
            sampling_params: 为 None ,将使用默认的采样参数, 
                当采样参数为单个值时,它将应用于每个提示(prompt)
                当采样参数为列表时, 该列表的长度必须与提示的数量相同, 并且会与提示逐一配对使用
            prompt_token_ids: 提示的token IDs列表. 如果为None, 将使用tokenizer将提示转换为token IDs
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if self.llm_runtime.model_config.skip_tokenizer_init and prompts is not None:
            raise ValueError("prompts must be None if skip_tokenizer_init "
                             "is True")
        if isinstance(prompts, str):
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")

        if prompts is not None:
            num_requests = len(prompts)
        else:
            assert prompt_token_ids is not None
            num_requests = len(prompt_token_ids)

        if sampling_params is None:
            sampling_params = SamplingParams()  # 使用默认采样参数

        elif isinstance(sampling_params,list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and sampling_params "
                             "must be the same.")
        if multi_modal_data:
            multi_modal_data.data = multi_modal_data.data.to(torch.float16)

        # 给推理引擎添加requests
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[i]
            self._add_request(
                prompt,
                sampling_params[i] if isinstance(sampling_params, list) else sampling_params,
                token_ids,
                lora_request=lora_request,
                multi_modal_data=MultiModalData(
                    type=multi_modal_data.type,
                    data=multi_modal_data.data[i].unsqueeze(0))
                if multi_modal_data else None,
            )
        return self._launch_runtime(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        request_id = str(next(self.request_counter)) # request id 从 0 开始计数
        self.llm_runtime.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids,
                                    lora_request=lora_request,
                                    multi_modal_data=multi_modal_data)

    def _launch_runtime(self, use_tqdm: bool) -> List[RequestOutput]:
        if use_tqdm:
            num_requests = self.llm_runtime.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests,
                        desc="Processed prompts",
                        dynamic_ncols=True)
        from sinfer.Intercept.mlu_definitions import check_latency_debug
        is_latency_debug = check_latency_debug()
        # 记录开始时间 
        if is_latency_debug:
            self.dump_info.capture_cpu_info()
            e2e_start_time = self.metric.get_mlu_cost_time()
        # 运行引擎
        outputs: List[RequestOutput] = []
        while self.llm_runtime.has_unfinished_requests():
            if is_latency_debug :
                self.dump_info.memory_usage() 
                start_time = self.metric.get_mlu_cost_time() 
            step_outputs = self.llm_runtime.step()     # iteration level
            if is_latency_debug:
                end_time = self.metric.get_mlu_cost_time()
                step_latency = end_time - start_time
                if self.metric.is_prefill_stage():
                    batch_size = len(step_outputs)
                self.metric.update_step_latency(step_latency)
                from sinfer.Intercept.mlu_definitions import check_latency_debug_with_device
                if check_latency_debug_with_device():
                    self.metric.update_step_latency_device(self.llm_runtime.get_latency())
                self.dump_info.memory_usage()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        # 计算延迟
        if is_latency_debug:
            e2e_end_time = self.metric.get_mlu_cost_time()
            e2e_latency = e2e_end_time - e2e_start_time
            self.metric.add_metrics(batch_size,e2e_latency)
        if use_tqdm:
            pbar.close()
        # 用request ID 排序 (有些请求可能比之前的请求更早完成)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
    def get_metrics(
        self,
        metrics_idx_start,
        only_average,
        input_len,
        output_len,
        tp_nums,
        quantization,
        dump_info=None
    ) -> None:
        '''
        @brief:该函数用来打印SInfer调用generate接口过程中代码统计的各项性能指标数据
        @params: 
            metrics_idx_start: 考虑存在调用generate接口为warmup过程的情况，
            因此设置该参数可忽略统计[0,metrics_idx_start)之间的数据,默认为0,即所有性能数据有效。
            only_average: True 只打印N次调用generate接口的平均性能 False 打印每次调用generate接口的性能及其均值 若N次性能数据波动较大，需自行排查测试环境是否稳定。
            其余参数:均为模型配置参数
        '''
        from sinfer.Intercept.mlu_definitions import check_latency_debug
        if check_latency_debug():
            self.metric.calc_metric(self.llm_runtime.model_config.model,
                                    self.llm_runtime.model_config.dtype,
                                    metrics_idx_start, only_average,
                                    input_len, output_len, tp_nums,
                                    quantization, dump_info)
        else:
            print("Warnning:please set SINFER_LATENCY_DEBUG=true!")