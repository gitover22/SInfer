from abc import ABC, abstractmethod
from typing import Callable, List

from transformers import PreTrainedTokenizer

from sinfer.task import Task, TaskOutput
from sinfer.config import SchedulerConfig
from sinfer.scheduler.TAScheduler import Scheduler
from sinfer.sys.output_processor.stop_checker import StopChecker
from sinfer.request import Request
from sinfer.transformers_utils.detokenizer import Detokenizer
from sinfer.utils import Counter

class TaskOutputProcessor(ABC):
    @staticmethod
    def create_output_processor(
        scheduler_config: SchedulerConfig,
        detokenizer: Detokenizer,
        scheduler: Scheduler,
        seq_counter: Counter,
        get_req_tokenizer: Callable[[Request], PreTrainedTokenizer],
        stop_checker: "StopChecker",
    ):
        if scheduler_config.num_lookahead_slots == 0:
            from sinfer.sys.output_processor.single_step import (
                SingleStepOutputProcessor)
            return SingleStepOutputProcessor(
                scheduler_config,
                detokenizer,
                scheduler,
                seq_counter,
                stop_checker,
            )
        else:
            from sinfer.sys.output_processor.multi_step import (
                MultiStepOutputProcessor)
            return MultiStepOutputProcessor(
                detokenizer,
                scheduler,
                seq_counter,
                get_req_tokenizer,
                stop_checker,
            )

    @abstractmethod
    def process_outputs(self, task: Task,
                        outputs: List[TaskOutput]) -> None:
        pass

    @abstractmethod
    def process_prompt_logprob(self, task: Task,
                               outputs: List[TaskOutput]) -> None:
        pass
