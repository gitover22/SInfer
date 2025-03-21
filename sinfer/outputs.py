import time
from typing import List, Optional, Union

from sinfer.task import Metadata, Task
from sinfer.lora.request import LoRARequest
from sinfer.request import (PromptLogprobs, SampleLogprobs,
                           RequestStatus)


class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[SampleLogprobs],
        finish_reason: Optional[str] = None,
        stop_reason: Union[int, str, None] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        self.lora_request = lora_request

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason}, "
                f"stop_reason={self.stop_reason})")


class RequestOutput:
    """The output data of a request.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: List[CompletionOutput],
        finished: bool,
        metrics: Optional[Metadata] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request

    @classmethod
    def from_task(cls, task: Task) -> "RequestOutput":
        reqs = task.get_reqs()
        if len(reqs) == 1:
            top_n_reqs = reqs
        else:
            # Get the top-n sequences.
            n = task.sampling_params.n
            if task.sampling_params.use_beam_search:
                sorting_key = lambda seq: seq.get_beam_search_score(
                    task.sampling_params.length_penalty)
            else:
                sorting_key = lambda seq: seq.get_cumulative_logprob()
            sorted_seqs = sorted(reqs, key=sorting_key, reverse=True)
            top_n_reqs = sorted_seqs[:n]

        # Create the outputs.
        # NOTE: We need omit logprobs here explicitly because the req
        # always has the logprobs of the sampled tokens even if the
        # logprobs are not requested.
        include_logprobs = task.sampling_params.logprobs is not None
        text_buffer_length = task.sampling_params.output_text_buffer_length
        outputs = [
            CompletionOutput(reqs.index(req),
                             req.get_output_text_to_return(text_buffer_length),
                             req.get_output_token_ids(),
                             req.get_cumulative_logprob(),
                             req.output_logprobs if include_logprobs else None,
                             RequestStatus.get_finished_reason(req.status),
                             req.stop_reason) for req in top_n_reqs
        ]

        # Every req in the task should have the same prompt.
        prompt = task.prompt
        prompt_token_ids = task.prompt_token_ids
        prompt_logprobs = task.prompt_logprobs
        finished = task.is_finished()
        finished_time = time.time() if finished else None
        task.set_finished_time(finished_time)
        return cls(task.task_id,
                   prompt,
                   prompt_token_ids,
                   prompt_logprobs,
                   outputs,
                   finished,
                   task.metadata,
                   lora_request=task.lora_request)

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"prompt_logprobs={self.prompt_logprobs}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished}, "
                f"metrics={self.metrics}, "
                f"lora_request={self.lora_request})")
