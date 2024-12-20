from dataclasses import dataclass
from typing import TYPE_CHECKING,Dict, List, Optional
import enum

from sinfer.lora.request import LoRARequest
from sinfer.request import PromptLogprobs, Request, RequestData, RequestMetrics, RequestOutput, RequestStatus
from sinfer.sampling_params import SamplingParams


if TYPE_CHECKING:
    import torch

class MultiModalData:
    """Multi modal request.

    Args:
        type: The data type.
        data: The actual data.
        The required shape and semantic meaning of it depends on the vision
        language config of the hosted model. 
        See `VisionLanguageConfig` in `config.py`.
    """

    class Type(enum.Enum):
        IMAGE = enum.auto()

    def __init__(self, type: Type, data: "torch.Tensor"):
        self.type = type
        self.data = data


@dataclass
class Task_State:
    """Mutable state tied to a specific sequence group"""

    # torch.Generator used in seeded sampling
    generator: Optional = None  # type: ignore


class Task:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: 请求id, 一个SG就是一个请求, 对应一个id
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
        lora_request: LoRA request.
        multi_modal_data: Multi modal data associated with the request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Request],
        sampling_params: SamplingParams,
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.metrics = RequestMetrics(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_queue=None)
        self.lora_request = lora_request
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.state = Task_State()
        self.multi_modal_data = multi_modal_data

    @property
    def prompt(self) -> str:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).data.prompt_token_ids

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def get_last_latency(self, now: float) -> Optional[float]:
        """Sets the last token time for Request level timings."""
        # If still in prefill phase, raise Error.
        if self.is_prefill():
            raise ValueError(
                "seq_group.get_last_latency() should not be called "
                "if the seq_group is in prefill phase.")

        # Otherwise return token latency.
        latency = now - self.metrics.last_token_time
        self.metrics.last_token_time = now
        return latency

    def maybe_set_first_token_time(self, time: float) -> None:
        """Sets the first token time for Request level timings."""
        # Note: in a case where a sequence_group is swapped and
        #   recomputed, the time between iterations is counted
        #   in TPOT, rather than recalculating TTFT (since from the )
        #   POV of the user, there is simply a long generation delay.
        if (self.metrics.first_token_time is None
                and self.get_seqs()[0].get_output_len() == 1):
            self.metrics.first_token_time = time

    def maybe_set_first_scheduled_time(self, time: float) -> None:
        """Sets the first scheduled time and time in queue for Request
        level timings."""
        if self.metrics.first_scheduled_time is None:
            self.metrics.first_scheduled_time = time
            self.metrics.time_in_queue = time - self.metrics.arrival_time

    def set_finished_time(self, time: Optional[float]) -> None:
        """Sets the finished time for Request level timings."""
        self.metrics.finished_time = time

    def get_max_num_running_seqs(self) -> int:
        """
        该seq_group在剩余生命周期内并行运行的最大seq数量
        剩余生命周期指从此刻一直到seq_group中所有的seq都做完推理
        用于估计若当前对一个seq_group做推理, 它将消耗多少gpu资源
        """
        if self.sampling_params.use_beam_search:
            # 若采用beam search，每1个推理阶段都是best_of（束宽）个seq在running
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_seqs():
                # 不适用束搜索时，best_of默认和n一致，即1个prompt产出n个outputs
                # 如果出现best_of > self.num_seqs()的情况，说明该seq_group刚从waiting变成running，准备做推理
                return self.sampling_params.best_of
            # 其余情况，返回未完成推理的seq数量
            return self.num_unfinished_seqs()

    def get_seqs(
        self,
        status: Optional[RequestStatus] = None,
    ) -> List[Request]:
        return list(self.seqs_dict.values()) if status is None else [
            seq for seq in self.seqs_dict.values() if seq.status == status
        ]

    def get_unfinished_seqs(self) -> List[Request]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Request]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        for seq in self.seqs_dict.values():
            if not seq.is_finished():
                seq.data.update_num_computed_tokens(num_new_computed_tokens)

    def get_num_uncomputed_tokens(self) -> int:
        num_uncomputed_tokens = 0
        for seq in self.get_seqs():
            if not seq.is_finished():
                num_uncomputed_tokens += seq.data.get_num_uncomputed_tokens()
        return num_uncomputed_tokens

    def num_seqs(self, status: Optional[RequestStatus] = None) -> int:
        # 获取序列组中的序列数量
        if status is None:
            return len(self.seqs_dict)

        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Request:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Request) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())

    def is_prefill(self) -> bool:
        # Every sequences should be in the same stage.
        return self.get_seqs()[0].is_prefill()

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs_dict)})")


class TaskMetadata:
    """Metadata for a sequence group. Used to create `AttentionMetadata`.

    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
        do_sample: True if sampling is required. Sampling is not required when
            e.g., prefill is chunked, and the current iteration only computes
            query tokens for prefill, we don't need sampling.
        token_chunk_size: The number of tokens to be processed (per sequence).
            None if chunking is not required.
        lora_request: LoRA request.
        computed_block_nums: The block numbers that are already computed,
            used in prefix caching.
        state: Internal state tied to this sequence group.
        multi_modal_data: Multi modal data.
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, RequestData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
        do_sample: bool = True,
        token_chunk_size: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
        computed_block_nums: Optional[List[int]] = None,
        state: Optional[Task_State] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables
        self.lora_request = lora_request
        self.computed_block_nums = computed_block_nums
        self.multi_modal_data = multi_modal_data
        self.state = Task_State() if state is None else state
        self._token_chunk_size = token_chunk_size
        self.do_sample = do_sample

        if self._token_chunk_size is None:
            if is_prompt:
                self._token_chunk_size = list(seq_data.values())[0].get_len()
            else:
                self._token_chunk_size = 1

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    @property
    def token_chunk_size(self) -> Optional[int]:
        """Return the number of tokens to be processed (chunk size)."""
        return self._token_chunk_size


class TaskOutput:
    """The model output associated with a sequence group."""

    def __init__(
        self,
        samples: List[RequestOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        # Prompt logprob for each prompt query token.
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"SequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


