import time
from tqdm import tqdm
from typing import Optional, List
from sinfer.entrypoints.llm import LLM
from sinfer.outputs import RequestOutput
from sinfer.Intercept.mlu_definitions import check_latency_debug, check_latency_debug_with_device
from sinfer.Intercept.intercept_utils import InterceptObject, set_batch_sizes_to_capture
from sinfer.Intercept.mlu_metric import LLMMetric
from sinfer.Intercept.dump_info import LLMDumpInfo
from sinfer.logger import init_logger


logger = init_logger(__name__)

LLM.metric = LLMMetric()
LLM.dump_info = LLMDumpInfo()

LLM__init__original = LLM.__init__

def intercept_LLM__init__(
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
    swap_space: int = 64,
    enforce_eager: bool = False,
    max_context_len_to_capture: Optional[int] = None,
    max_seq_len_to_capture: int = 8192,
    disable_custom_all_reduce: bool = False,
    **kwargs,
) -> None:
    '''
    支持用户定义最大捕获batch大小, 并初始化LLMDumpInfo
    '''
    if "max_batch_size_to_capture" in kwargs:
        max_batch_size_to_capture = kwargs.pop("max_batch_size_to_capture")
        if max_batch_size_to_capture:
            set_batch_sizes_to_capture(max_batch_size_to_capture)

    LLM.dump_info.init_param(tensor_parallel_size=tensor_parallel_size, dtype=dtype, 
               kv_cache_dtype=kwargs.get('kv_cache_dtype', 'default_value'),
               quantization=quantization,
               model=model, trust_remote_code=kwargs.get('trust_remote_code', 'default_value'))
    
    # 调用原来的构造函数
    LLM__init__original(
        self=self,
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
        **kwargs
    )
    
    LLM.dump_info.memory_usage()
    

def intercept_LLM_get_metrics(
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
    if check_latency_debug():
        self.metric.calc_metric(self.llm_runtime.model_config.model,
                                self.llm_runtime.model_config.dtype,
                                metrics_idx_start, only_average,
                                input_len, output_len, tp_nums,
                                quantization, dump_info)
    else:
        print("Warnning:please set SINFER_LATENCY_DEBUG=true!")


def intercept_LLM_launch_runtime(self, use_tqdm: bool) -> List[RequestOutput]:
    if use_tqdm:
        num_requests = self.llm_runtime.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests,
                    desc="Processed prompts",
                    dynamic_ncols=True)
    
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
        step_outputs = self.llm_runtime.step() # 一个step
       
        if is_latency_debug:
            end_time = self.metric.get_mlu_cost_time()
            step_latency = end_time - start_time
            if self.metric.is_prefill_stage():
                batch_size = len(step_outputs)
            self.metric.update_step_latency(step_latency)
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
    # 按照request ID 排序
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    return outputs

InterceptObject.add_intercept(LLM, LLM.__init__, intercept_LLM__init__)
InterceptObject.add_intercept(LLM,"get_metrics", intercept_LLM_get_metrics)
InterceptObject.add_intercept(LLM,LLM._launch_runtime, intercept_LLM_launch_runtime)