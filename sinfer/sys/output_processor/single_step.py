from typing import Dict, List, Tuple, Union

from sinfer.task import Task, TaskOutput
from sinfer.config import SchedulerConfig
from sinfer.scheduler.TAScheduler import Scheduler
from sinfer.sys.output_processor.interfaces import (
    TaskOutputProcessor)
from sinfer.sys.output_processor.stop_checker import StopChecker
from sinfer.logger import init_logger
from sinfer.sampling_params import SamplingParams
from sinfer.request import (Request, RequestOutput, RequestStatus)
from sinfer.transformers_utils.detokenizer import Detokenizer
from sinfer.utils import Counter

logger = init_logger(__name__)


class SingleStepOutputProcessor(TaskOutputProcessor):
    """该逻辑发生在模型返回生成的token ids之后，在调度下一个批次之前
    SingleStepOutputProcessor 专门用于模型每次调用最多生成一个token的情况
    这排除了推测解码或多步解码等配置。这使得束搜索采样成为可能，
    """
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        detokenizer: Detokenizer,
        scheduler: Scheduler,
        seq_counter: Counter,
        stop_checker: StopChecker,
    ):
        self.scheduler_config = scheduler_config
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.stop_checker = stop_checker

    def process_outputs(self, task: Task,
                        outputs: List[TaskOutput]) -> None:
        """Append all new tokens to requests in the task. Fork any
        surviving beam candidates; free any unsurviving ones.
        """
        assert (len(outputs) == 1
                ), f"{type(self)} does not support multiple outputs per step"
        return self._process_task_outputs(task, outputs[0])

    def process_prompt_logprob(self, task: Task,
                               outputs: List[TaskOutput]) -> None:
        assert len(outputs) == 1, ("Single step should only has 1 output.")
        output = outputs[0]
        prompt_logprobs = output.prompt_logprobs
        if (prompt_logprobs is not None
                and task.sampling_params.detokenize and self.detokenizer):
            self.detokenizer.decode_prompt_logprobs_inplace(
                task, prompt_logprobs)
            if not task.prompt_logprobs:
                # 第一个提示token的logprob为None，因为它没有前置tokens
                task.prompt_logprobs = [None]
            task.prompt_logprobs.extend(prompt_logprobs)

    def _process_task_outputs(self, task: Task,
                                        outputs: TaskOutput) -> None:
        # Process samples
        samples = outputs.samples
        parent_seqs = task.get_reqs(status=RequestStatus.RUNNING)
        existing_finished_seqs = task.get_finished_reqs()
        parent_child_dict: Dict[int, List[RequestOutput]] = {
            parent_seq.req_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_req_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Request, Request]] = []

        # Process the child samples for each parent request
        for parent in parent_seqs:
            child_samples: List[RequestOutput] = parent_child_dict[
                parent.req_id]
            if len(child_samples) == 0:
                # This parent request has no children samples. Remove
                # the parent request from the task since it will
                # not be used in the future iterations.
                parent.status = RequestStatus.FINISHED_ABORTED
                task.remove(parent.req_id)
                self.scheduler.free_req(parent)
                continue
            # Fork the parent request if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id: int = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token,
                                      child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent request for the last child sample.
            # We reuse the parent request here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token,
                                   last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            if task.sampling_params.detokenize and self.detokenizer:
                new_char_count = self.detokenizer.decode_sequence_inplace(
                    seq, task.sampling_params)
            else:
                new_char_count = 0
            self.stop_checker.maybe_stop_request(seq, new_char_count,
                                                  task.sampling_params)

        # Non-beam search case
        if not task.sampling_params.use_beam_search:
            # For newly created child requests, add them to the task
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    task.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_req(parent, seq)

            # Free the finished and selected parent requests' memory in block
            # manager. Keep them in the task as candidate output.
            # NOTE: we need to fork the new requests before freeing the
            # old requests.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_req(seq)
            return

        # Beam search case
        # Select the child requests to keep in the task.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = task.sampling_params.best_of
        length_penalty = task.sampling_params.length_penalty

        # Select the newly finished requests with the highest scores
        # to replace existing finished requests.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False)
                                  for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                             if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished requests by their scores.
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=x[0].eos_token_id),
                               reverse=True)
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child request finishes and has a high
                # score, so we will add it into the task.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child request finishes but has a low
                # score, so we will not add it into the task.
                # Additionally, if this request is a continuation of a
                # parent request, we will need remove the parent request
                # from the task.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished request has a low score, so we will
                # remove it from the task.
                task.remove(seq.req_id)

        # select the top beam_width requests from the running
        # requests for the next iteration to continue the beam
        # search.
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                              if not seq.is_finished()]
        # Sort the running requests by their scores.
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=x[0].eos_token_id),
                                reverse=True)

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running requests, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished requests, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                task.sampling_params.early_stopping,
                task.sampling_params, best_running_seq, current_worst_seq)

        if stop_beam_search:
            # Stop the beam search and remove all the running requests from
            # the task.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width requests
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running requests will not be used in the next
            # iteration. Again, if these requests are continuations of
            # parent requests, we will need to remove the parent requests
            # from the task.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child requests, add them to the task
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                task.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_req(parent, seq)

        # Free the finished and selected parent requests' memory in block
        # manager. Keep them in the task as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_req(seq)

        # Remove the unselected parent requests from the task and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent request if it is not selected for next
                # iteration
                task.remove(seq.seq_id)
                self.scheduler.free_req(seq)

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Request,
        current_worst_seq: Request,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=current_worst_seq.eos_token_id)
        if early_stopping is False:
            highest_attainable_score = best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=best_running_seq.eos_token_id)
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # requests. The highest attainable score calculation is
                # based on the longest possible request length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=best_running_seq.eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter reqs. The
                # highest attainable score calculation is based on the current
                # request length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=best_running_seq.eos_token_id))
        return current_worst_score >= highest_attainable_score
