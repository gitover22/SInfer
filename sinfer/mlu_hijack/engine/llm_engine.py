import time
from typing import Optional, List
from sinfer.engine.llm_engine import LLMEngine
from sinfer.mlu_hijack.mlu_definitions import USE_PAGED, BlockSizeInfo
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.sampling_params import SamplingParams
from sinfer.sequence import MultiModalData, Sequence, SequenceGroup
from sinfer.lora.request import LoRARequest
from sinfer.logger import init_logger

logger = init_logger(__name__)


sinfer_engine__llm_engine__LLMEngine__add_request_org = LLMEngine.add_request


def sinfer_engine__llm_engine__LLMEngine__add_request(
    self,
    request_id: str,
    prompt: Optional[str],
    sampling_params: SamplingParams,
    prompt_token_ids: Optional[List[int]] = None,
    arrival_time: Optional[float] = None,
    lora_request: Optional[LoRARequest] = None,
    multi_modal_data: Optional[MultiModalData] = None,
) -> None:
    """Add a request to the engine's request pool.

    The request is added to the request pool and will be processed by the
    scheduler as `engine.step()` is called. The exact scheduling policy is
    determined by the scheduler.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string. Can be None if prompt_token_ids is
            provided.
        sampling_params: The sampling parameters for text generation.
        prompt_token_ids: The token IDs of the prompt. If None, we
            use the tokenizer to convert the prompts to token IDs.
        arrival_time: The arrival time of the request. If None, we use
            the current monotonic time.
        multi_modal_data: Multi modal data per request.

    Details:
        - Set arrival_time to the current time if it is None.
        - Set prompt_token_ids to the encoded prompt if it is None.
        - Create `best_of` number of :class:`~sinfer.Sequence` objects.
        - Create a :class:`~sinfer.SequenceGroup` object
            from the list of :class:`~sinfer.Sequence`.
        - Add the :class:`~sinfer.SequenceGroup` object to the scheduler.

    Example:
        >>> # initialize engine
        >>> engine = LLMEngine.from_engine_args(engine_args)
        >>> # set request arguments
        >>> example_prompt = "Who is the president of the United States?"
        >>> sampling_params = SamplingParams(temperature=0.0)
        >>> request_id = 0
        >>>
        >>> # add the request to the engine
        >>> engine.add_request(
        >>>    str(request_id),
        >>>    example_prompt,
        >>>    SamplingParams(temperature=0.0))
        >>> # continue the request processing
        >>> ...
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
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        lora_request=lora_request)

    # Create the sequences.
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
    # Add the eos token id into the sampling_params to support min_tokens
    # processing
    if seq.eos_token_id is not None:
        sampling_params.all_stop_token_ids.add(seq.eos_token_id)
    sampling_params.update_from_generation_config(
        self.generation_config_fields)

    '''
    =============================
    Added by sinfer_mlu
    =============================
    @brief: check input_len + output_len <= block_size
    '''
    num_prompt_tokens = 0
    out_len = sampling_params.max_tokens
    if (USE_PAGED == False and BlockSizeInfo.BLOCK_SIZE < out_len + seq.get_len()):
        raise ValueError("BLOCK_SIZE(" + str(BlockSizeInfo.BLOCK_SIZE) +
            ") can't smaller than input_len(" + str(seq.get_len()) + ") + output_len(" + str(out_len) + ").")
    '''
    ==================
    End of modification
    ==================
    '''

    # Create the sequence group.
    seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                arrival_time, lora_request, multi_modal_data)

    # Add the sequence group to the scheduler.
    self.scheduler.add_seq_group(seq_group)


def sinfer__engine__llm_engine__LLMEngine__get_latency(self):
    latency = self.model_executor.get_latency()
    return latency


MluHijackObject.add_hijack_object(LLMEngine,
                                  LLMEngine.add_request,
                                  sinfer_engine__llm_engine__LLMEngine__add_request)
MluHijackObject.add_hijack_object(LLMEngine,
                                  "get_latency",
                                  sinfer__engine__llm_engine__LLMEngine__get_latency)