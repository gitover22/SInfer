from typing import Optional
from sinfer.entrypoints.llm import LLM
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject, set_batch_sizes_to_capture
from sinfer.sys.arg_utils import RuntimeArgs
from sinfer.sys.runtime import LLMRuntime
from sinfer.usage.usage_lib import UsageContext
from sinfer.utils import Counter


def _LLM_init_(
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
    # Support user to define the max capture batch size
    if "max_batch_size_to_capture" in kwargs:
        max_batch_size_to_capture = kwargs.pop("max_batch_size_to_capture")
        if max_batch_size_to_capture:
            set_batch_sizes_to_capture(max_batch_size_to_capture)
       
    # pop context_parallel_size
    context_parallel_size = 1
    if "context_parallel_size" in kwargs:
        context_parallel_size = kwargs.pop("context_parallel_size")

    if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
    engine_args = RuntimeArgs(
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

    # set context_parallel_size to engine_args
    setattr(engine_args, 'context_parallel_size', context_parallel_size)
    
    self.llm_engine = LLMRuntime.from_engine_args(
        engine_args, usage_context=UsageContext.LLM_CLASS)
    self.request_counter = Counter()

MluHijackObject.add_hijack_object(LLM,
                                  LLM.__init__,
                                  _LLM_init_)