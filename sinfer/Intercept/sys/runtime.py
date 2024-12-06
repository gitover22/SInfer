import time
from typing import Optional, List
from sinfer.sys.runtime import LLMRuntime
from sinfer.Intercept.mlu_definitions import USE_PAGED, BlockSizeInfo
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.sampling_params import SamplingParams
from sinfer.sequence import MultiModalData, Sequence, SequenceGroup
from sinfer.lora.request import LoRARequest
from sinfer.logger import init_logger

logger = init_logger(__name__)


add_request_org = LLMRuntime.add_request


def intercept_add_request(
    self,
    request_id: str,
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
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        lora_request=lora_request)

    block_size = self.cache_config.block_size
    seq_id = next(self.seq_counter)
    eos_token_id = None
    if self.tokenizer:
        eos_token_id = self.tokenizer.get_lora_tokenizer(
            lora_request).eos_token_id
    else:
        logger.warning("Use None for EOS token id because tokenizer is "
                        "not initialized")
    seq = Sequence(seq_id, prompt, prompt_token_ids, block_size, 
                    eos_token_id, lora_request)

    # Defensive copy of SamplingParams, which are used by the sampler,
    # this doesn't deep-copy LogitsProcessor objects
    sampling_params = sampling_params.clone()
    # Add the eos token id into the sampling_params to support min_tokens processing
    if seq.eos_token_id is not None:
        sampling_params.all_stop_token_ids.add(seq.eos_token_id)
    sampling_params.update_from_generation_config(
        self.generation_config_fields)

   
    # 检查 input_len + output_len <= block_size(非分页模式下，缓存的块大小需要能够容纳输入和输出的token)
    num_prompt_tokens = 0
    out_len = sampling_params.max_tokens # 生成的最大token数量
    if (USE_PAGED == False and BlockSizeInfo.BLOCK_SIZE < out_len + seq.get_len()):
        raise ValueError("BLOCK_SIZE(" + str(BlockSizeInfo.BLOCK_SIZE) +
            ") can't smaller than input_len(" + str(seq.get_len()) + ") + output_len(" + str(out_len) + ").")

    # 创建SG
    seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                arrival_time, lora_request, multi_modal_data)

    # 添加SG到调度器waiting队列.
    self.scheduler.add_seq_group(seq_group)


def intercept_get_latency(self):
    latency = self.model_executor.get_latency()
    return latency


InterceptObject.add_intercept(LLMRuntime,
                                  LLMRuntime.add_request,
                                  intercept_add_request)
InterceptObject.add_intercept(LLMRuntime,
                                  "get_latency",
                                  intercept_get_latency)