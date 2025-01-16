import functools
from typing import Callable, List

from transformers import PreTrainedTokenizer

from sinfer.task import Task, TaskOutput
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


class MultiStepOutputProcessor(TaskOutputProcessor):
    """TaskOutputProcessor 处理与去标记化和停止条件相关的逻辑。它专门用于“多步解码”，
    在这种解码方式中，SInfer的工作者可能每次调用生成多个tokens。
    目前，这与束搜索等高级采样技术是互斥的，这也促使将该逻辑与单步输出处理器分离。

    该类负责处理一些任务，例如正确地将所有新token ids附加到它们的序列中，去标记化新token ids，
    在生成eos token后截断新的输出tokens，以及正确处理在单个批次中每个序列生成的新输出token数量不同的情况。
    """

    def __init__(
        self,
        detokenizer: Detokenizer,
        scheduler: Scheduler,
        seq_counter: Counter,
        get_req_tokenizer: Callable[[Request], PreTrainedTokenizer],
        stop_checker: StopChecker,
    ):
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.get_req_tokenizer = get_req_tokenizer
        self.stop_checker = stop_checker

    def process_prompt_logprob(self, task: Task,
                               outputs: List[TaskOutput]) -> None:
        self._log_prompt_logprob_unsupported_warning_once()

    @staticmethod
    @functools.lru_cache()
    def _log_prompt_logprob_unsupported_warning_once():
        logger.warning(
            "Prompt logprob is not supported by multi step workers. "
            "(e.g., speculative decode uses multi step workers).")

    def process_outputs(self, task: Task,
                        outputs: List[TaskOutput]) -> None:
        """将输出中的新tokens附加到任务中的请求。

        这仅支持大小为1的任务。它支持每个请求生成多个新token。

        该过程应用了诸如停止条件检查和去标记化的逻辑，
        包括释放已完成的请求。它还处理在eos token后生成tokens的情况。
        """
        reqs = task.get_reqs(status=RequestStatus.RUNNING)

        assert reqs, "expected running requests"
        assert len(reqs) == 1, (
            "Beam search not supported in multi-step decoding.")
        req = reqs[0]

        # 由于每个任务只有一个请求，我们可以获取第一个样本
        samples = [outputs[step].samples[0] for step in range(len(outputs))]

        # -1表示输出token无效（例如，由于推测解码拒绝tokens）
        valid_samples = [
            sample for sample in samples if sample.output_token != -1
        ]
        assert valid_samples

        self._process_req_outputs(req, valid_samples,
                                  task.sampling_params)

    def _process_req_outputs(self, req: Request,
                             valid_samples: List[RequestOutput],
                             sampling_params: SamplingParams) -> None:
        output_token_ids = [sample.output_token for sample in valid_samples]
        output_logprobs = [sample.logprobs for sample in valid_samples]

        # 如有必要，截断至max_tokens
        remaining_tokens = sampling_params.max_tokens - (req.get_output_len() +
                                                         len(output_token_ids))
        if remaining_tokens < 0:
            valid_samples = valid_samples[:remaining_tokens]
            output_token_ids = output_token_ids[:remaining_tokens]

        # 截断eos后的任何token。由于推测解码
        # 会生成固定数量的tokens，而不在块内评估停止条件
        # 这可能导致eos token被无意忽略
        if not sampling_params.ignore_eos:
            eos_token_id = self.get_req_tokenizer(req).eos_token_id
            # 避免在正常路径中调用.index，因异常处理开销较大
            for i in range(len(output_token_ids)):
                if output_token_ids[i] == eos_token_id:
                    output_token_ids = output_token_ids[:i + 1]
                    valid_samples = valid_samples[:i + 1]
                    break

        # 逐步将token附加到请求中，就像我们每次只生成一个新token一样
        for output_token_id, output_logprob in zip(output_token_ids,
                                                   output_logprobs):
            req.append_token_id(
                token_id=output_token_id,
                logprobs=output_logprob,
            )

            new_char_count = 0
            if sampling_params.detokenize:
                new_char_count = self.detokenizer.decode_sequence_inplace(
                    req, sampling_params)

            self.stop_checker.maybe_stop_request(
                req,
                new_char_count=new_char_count,
                sampling_params=sampling_params)
            if req.is_finished():
                break

        if req.is_finished():
            self.scheduler.free_req(req)
