import copy
import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from sinfer.block import LogicalTokenBlock
from sinfer.lora.request import LoRARequest

if TYPE_CHECKING:
    import torch

@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


# {token_id -> logprob} per each task. None if the corresponding
# task doesn't require prompt logprob.
PromptLogprobs = List[Optional[Dict[int, Logprob]]]
# {token_id -> logprob} for each task.
SampleLogprobs = List[Dict[int, Logprob]]


class RequestStatus(enum.Enum):
    # 请求的状态
    READY = enum.auto()
    RUNNING = enum.auto()
    BLOCKED = enum.auto()
    # 完成状态
    FINISHED_STOPPED = enum.auto()  # 正常完成 eos
    FINISHED_LENGTH_CAPPED = enum.auto() # seq的长度达到最大长度限制，而结束推理
    FINISHED_ABORTED = enum.auto() # 不正常状态，而被终止的推理
    FINISHED_IGNORED = enum.auto() # 因prompt过长而被终止执行的推理(忽略)

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        if status == RequestStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == RequestStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == RequestStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == RequestStatus.FINISHED_IGNORED:
            # 请求的prompt长度大于模型的最大文本长度.
            # 完成的原因应该是"length"(兼容OpenAI API)
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class Request_Stage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


class RequestData:
    """
        请求的数据,包括prompt和输出的token ids,以及累积的对数概率
    Args:
        prompt_token_ids: prompt对应的token ids
        output_token_ids: 输出对应的token IDs.
        cumulative_logprob: 输出的累积对数概率
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
        output_token_ids: Optional[List[int]] = None,
    ) -> None:
        if output_token_ids is None:
            output_token_ids = []

        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = output_token_ids
        self.cumulative_logprob = 0.0
        self._num_computed_tokens = 0 # 已经计算的预填充（prefill）令牌的数量
        self._stage: Request_Stage = Request_Stage.PREFILL

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_num_computed_tokens(self) -> int:
        return self._num_computed_tokens


    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        '''更新已经计算的令牌数量'''
        self._num_computed_tokens += num_new_computed_tokens
        assert self._num_computed_tokens <= self.get_len(), (
            self._num_computed_tokens, self.get_len())
        # 如果所有令牌都已经计算完毕，进入decode阶段
        if self.get_num_uncomputed_tokens() == 0:
            self._stage = Request_Stage.DECODE

    def reset_state_for_recompute(self) -> None:
        """Reset the number of computed tokens from this sequence. It is
        supposed to be called when a sequence needs to be started from
        the beginning again (e.g., sequence is preempted).
        """
        self._num_computed_tokens = 0
        self._stage = Request_Stage.PREFILL

    def get_num_uncomputed_tokens(self) -> int:
        """返回没被计算的prefill token数量"""
        # 使用 get_len 方法而不是单独使用提示令牌的长度（prompt_len），
        # 是因为在重新计算（recompute）过程中，需要同时预填充提示和输出令牌。这确保了在重新计算时，所有相关的令牌都被考虑在内
        return self.get_len() - self.get_num_computed_tokens()

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def get_prompt_token_ids(self) -> List[int]:
        return self.prompt_token_ids

    def get_output_token_ids(self) -> List[int]:
        return self.output_token_ids

    @property
    def stage(self) -> Request_Stage:
        return self._stage

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")


class Request:
    def __init__(
        self,
        req_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.req_id = req_id
        self.prompt = prompt
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request

        self.data: RequestData = RequestData(prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = [] # 每个req都会维护自己的逻辑块列表
        # 使用提示(prompt)的token ID初始化逻辑token块。
        self._append_tokens_to_blocks(prompt_token_ids)
        self.status = RequestStatus.READY
        self.stop_reason: Union[int, str, None] = None

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def get_output_text_to_return(self, buffer_length: int):
        # We return the full output text if the sequence is finished.
        truncate = buffer_length and not self.is_finished()
        return self.output_text[:-buffer_length] if truncate else (
            self.output_text)

    def hash_of_block(self, logical_idx: int) -> int:
        # TODO This can produce incorrect hash when block size > prompt size

        # Compute the number of tokens in the sequence
        # TODO: The current hashing function is O(L^2). We should optimize this in the future.
        num_tokens = self.num_hashed_tokens_of_block(logical_idx)
        return hash(
            (tuple(self.data.get_token_ids()[0:num_tokens]), self.lora_int_id))

    def num_hashed_tokens_of_block(self, logical_idx: int):
        """
        计算截止到序号为logical_idx的逻辑块为止（包含这个逻辑块），一共有多少个token
        例如block_size = 16, logical_idx = 1,
        则截止到序号为1的逻辑块为止，一共有 1*16 + 16 = 32个逻辑块
        """
        return logical_idx * self.block_size + self.block_size

    def reset_state_for_recompute(self):
        """Reset the sequence states for recomputation."""
        self.data.reset_state_for_recompute()

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        """
        将token_ids动态填入逻辑块列表中
        Args:
            token_ids: prompt部分的token_ids
        """
        cursor = 0
        # 遍历prompt token_ids中的每一个token_id
        while cursor < len(token_ids):
            if not self.logical_token_blocks: # 如果当前逻辑块列表（logical_token_blocks）为空
                self._append_logical_block() # 则先append一个逻辑块，该逻辑块index为0，大小为16，其中的每一个token_id为-1

            last_block = self.logical_token_blocks[-1] # 取出逻辑块列表中的最后一个逻辑块
            if last_block.is_full(): # 如果这最后一个逻辑块中已经没有槽位
                self._append_logical_block() # 那么再append一个空的逻辑块
                last_block = self.logical_token_blocks[-1] # 把这个新append的逻辑块取出来

            num_empty_slots = last_block.get_num_empty_slots()  # 检查当前取出的逻辑块中空槽位的数量
            last_block.append_tokens(token_ids[cursor:cursor +
                                               num_empty_slots])
            cursor += num_empty_slots

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, Logprob],
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id].logprob)

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_prompt_token_ids(self) -> List[int]:
        return self.data.get_prompt_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 1.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[int] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # NOTE: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def fork(self, new_req_id: int) -> "Request":
        new_req = copy.deepcopy(self)
        new_req.req_id = new_req_id
        return new_req

    def get_num_new_tokens(self) -> int:
        """Get the number of new tokens to be computed.

        Returns:
            The new number of tokens to be computed. I.e., 1 for decode, or
            the remaining prompt size for prefill.
        """
        if self.data.stage == Request_Stage.DECODE:
            return 1
        return self.data.get_num_uncomputed_tokens()

    def is_prefill(self) -> bool:
        return self.data.stage == Request_Stage.PREFILL

    def __repr__(self) -> str:
        return (f"Request(req_id={self.req_id}, "
                f"status={self.status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


class RequestOutput:
    """The model output associated with a request.

    Args:
        parent_req_id: The ID of the parent req (for forking in beam search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        parent_req_id: int,
        output_token: int,
        logprobs: Dict[int, Logprob],
    ) -> None:
        self.parent_req_id = parent_req_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"RequestOutput(parent_req_id={self.parent_req_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RequestOutput):
            raise NotImplementedError()
        equal = (self.parent_req_id == other.parent_req_id
                 and self.output_token == other.output_token)
        log_probs_equal = other.logprobs == self.logprobs
        return equal and log_probs_equal


@dataclass
class SamplerOutput:
    """For each task, we generate a list of SequenceOutput object,
    each of which contains one possible candidate for the next token.

    This datastructure implements methods so it can be used like a list, but
    also has optional fields for device tensors.
    """
    from sinfer.task import TaskOutput
    outputs: List[TaskOutput]

    # On-device tensor containing probabilities of each token.
    sampled_token_probs: Optional["torch.Tensor"] = None

    # On-device tensor containing the logprobs of each token.
    logprobs: Optional["torch.Tensor"] = None

    # On-device tensor containing the sampled token ids.
    sampled_token_ids: Optional["torch.Tensor"] = None

    def __getitem__(self, idx: int):
        return self.outputs[idx]

    def __setitem__(self, idx: int, value):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other,
                          self.__class__) and self.outputs == other.outputs

    def __repr__(self) -> str:
        """Show the shape of a tensor instead of its values to reduce noise.
        """
        sampled_token_probs_repr = ("None" if self.sampled_token_probs is None
                                    else self.sampled_token_probs.shape)
        sampled_token_ids_repr = ("None" if self.sampled_token_ids is None else
                                  self.sampled_token_ids.shape)
        return (
            f"SamplerOutput(outputs={self.outputs}, "
            f"sampled_token_probs={sampled_token_probs_repr}, "
            f"sampled_token_ids={sampled_token_ids_repr}, "
            f"spec_decode_worker_metrics={self.spec_decode_worker_metrics})")


@dataclass
class ExecuteModelRequest:
    from sinfer.task import TaskMetadata
    """The model execution request."""
    # The sequence group metadata list.
    task_metadata_list: List[TaskMetadata]
    # Blocks to swap in. Dict of CPU -> GPU block number.
    blocks_to_swap_in: Dict[int, int] = field(default_factory=dict)
    # Blocks to swap out. Dict of GPU -> CPU block number.
    blocks_to_swap_out: Dict[int, int] = field(default_factory=dict)
    # Blocks to copy. Source to a list of dest blocks.
    blocks_to_copy: Dict[int, List[int]] = field(default_factory=dict)
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int = 0
    # The number of requests in the running queue.
    running_queue_size: int = 0

    def clone(
        self, task_metadata_list: List[TaskMetadata]
    ) -> "ExecuteModelRequest":
        """Clone the request with a new task metadata list."""
        return ExecuteModelRequest(
            task_metadata_list=task_metadata_list,
            blocks_to_swap_in=self.blocks_to_swap_in.copy(),
            blocks_to_swap_out=self.blocks_to_swap_out.copy(),
            blocks_to_copy=self.blocks_to_copy.copy(),
            num_lookahead_slots=self.num_lookahead_slots,
            running_queue_size=self.running_queue_size,
        )
