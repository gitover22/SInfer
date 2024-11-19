from typing import List, Optional, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sinfer.engine.arg_utils import EngineArgs
from sinfer.engine.llm_engine import LLMEngine
from sinfer.lora.request import LoRARequest
from sinfer.outputs import RequestOutput
from sinfer.sampling_params import SamplingParams
from sinfer.sequence import MultiModalData
from sinfer.usage.usage_lib import UsageContext
from sinfer.utils import Counter


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", "squeezellm", "weightonly", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use `max_seq_len_to_capture` instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig
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
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
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
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer.tokenizer = tokenizer

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
            prompt_token_ids: 提示的token IDs列表. 如果为None, 我们将使用tokenizer将提示转换为token IDs
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if self.llm_engine.model_config.skip_tokenizer_init \
            and prompts is not None:
            raise ValueError("prompts must be None if skip_tokenizer_init "
                             "is True")
        if isinstance(prompts, str):
            # to list
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

        elif isinstance(sampling_params,
                        list) and len(sampling_params) != num_requests:
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
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids,
                                    lora_request=lora_request,
                                    multi_modal_data=multi_modal_data)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests,
                        desc="Processed prompts",
                        dynamic_ncols=True)
        # 运行引擎
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # 用request ID 排序 (有些请求可能比之前的请求更早完成)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs