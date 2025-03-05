from dataclasses import dataclass
from typing import TYPE_CHECKING,Dict, List, Optional
import enum
import os 
from sinfer.lora.request import LoRARequest
from sinfer.request import PromptLogprobs, Request, RequestData, RequestOutput, RequestStatus
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
class Metadata:
    request_complexity: Optional[float] = None  # 请求复杂度
    resource_requirements: Optional[dict] = None  # 资源需求
    priority: int = 0  # 用户优先级（0 表示低优先级，1 表示高优先级）
    deadline: Optional[float] = None  # 请求截止时间
    arrival_time: float = 0.0 # 请求到达的时间
    last_token_time: float = 0.0
    first_scheduled_time: Optional[float] = None # 请求第一次被调度的时间
    first_token_time: Optional[float] = None # 第一个令牌生成的时间
    time_in_pool: Optional[float] = None # 请求在池中等待的时间
    finished_time: Optional[float] = None # 请求完成的时间
    prompt_length: Optional[int] = None # 请求的提示长度
    predict_length: Optional[int] = None # 预测长度

@dataclass
class Task_State:
    """Mutable state tied to a specific task"""

    # torch.Generator used in seeded sampling
    generator: Optional = None  # type: ignore




class Task:
    """A group of reqs that are generated from the same prompt.

    Args:
        task_id: 一个Task对应一个id
        reqs: The list of requests
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
        lora_request: LoRA request.
        multi_modal_data: Multi modal data associated with the request.
    """

    def __init__(
        self,
        task_id: str,
        reqs: List[Request],
        sampling_params: SamplingParams,
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        self.task_id = task_id
        self.reqs_dict = {req.req_id: req for req in reqs}
        self.sampling_params = sampling_params
        self.metadata = Metadata(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_pool=None)
        self.metadata.prompt_length = max((len(req.prompt) for req in reqs if req.prompt is not None), default=0)
        # predictor
        if os.getenv("PREDICTOR", "False").lower() == "true":
            from sinfer.predictor.predictor import LengthPredictor
            predictor = LengthPredictor(model_path='/home/huafeng/SInfer/sinfer/predictor/distilbert-based-uncased-finetuned')
            self.metadata.predict_length = predictor.predict_length(reqs[0].prompt)
        self.lora_request = lora_request
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.state = Task_State()
        self.multi_modal_data = multi_modal_data

    @property
    def prompt(self) -> str:
        # All requests in the group should have the same prompt.
        # We use the prompt of an arbitrary request.
        return next(iter(self.reqs_dict.values())).prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        # All requests in the group should have the same prompt.
        # We use the prompt of an arbitrary request.
        return next(iter(self.reqs_dict.values())).data.prompt_token_ids

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def get_last_latency(self, now: float) -> Optional[float]:
        """Sets the last token time for Request level timings."""
        # If still in prefill phase, raise Error.
        if self.is_prefill():
            raise ValueError(
                "task.get_last_latency() should not be called "
                "if the task is in prefill phase.")

        # Otherwise return token latency.
        latency = now - self.metadata.last_token_time
        self.metadata.last_token_time = now
        return latency

    def maybe_set_first_token_time(self, time: float) -> None:
        """Sets the first token time for Request level timings."""
        # Note: in a case where a req_group is swapped and
        #   recomputed, the time between iterations is counted
        #   in TPOT, rather than recalculating TTFT (since from the )
        #   POV of the user, there is simply a long generation delay.
        if (self.metadata.first_token_time is None
                and self.get_reqs()[0].get_output_len() == 1):
            self.metadata.first_token_time = time

    def maybe_set_first_scheduled_time(self, time: float) -> None:
        """Sets the first scheduled time and time in queue for Request
        level timings."""
        if self.metadata.first_scheduled_time is None:
            self.metadata.first_scheduled_time = time
            self.metadata.time_in_pool = time - self.metadata.arrival_time

    def set_finished_time(self, time: Optional[float]) -> None:
        """Sets the finished time for Request level timings."""
        self.metadata.finished_time = time

    def get_task_running_reqs(self) -> int:
        """
        该task在剩余生命周期内并行运行的最大req数量
        剩余生命周期指从此刻一直到task中所有的req都做完推理
        用于估计若当前对一个task做推理, 它将消耗多少mlu资源
        """
        if self.sampling_params.use_beam_search:
            # 若采用beam search，每1个推理阶段都是best_of（束宽）个req在running
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_reqs():
                # 不适用束搜索时，best_of默认和n一致，即1个prompt产出n个outputs
                # 如果出现best_of > self.num_reqs()的情况，说明该task刚从ready变成running，准备做推理
                return self.sampling_params.best_of
            # 其余情况，返回未完成推理的seq数量
            return self.num_unfinished_reqs()

    def get_reqs(
        self,
        status: Optional[RequestStatus] = None,
    ) -> List[Request]:
        return list(self.reqs_dict.values()) if status is None else [
            req for req in self.reqs_dict.values() if req.status == status
        ]

    def get_unfinished_reqs(self) -> List[Request]:
        return [
            req for req in self.reqs_dict.values() if not req.is_finished()
        ]

    def get_finished_reqs(self) -> List[Request]:
        return [req for req in self.reqs_dict.values() if req.is_finished()]

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        for req in self.reqs_dict.values():
            if not req.is_finished():
                req.data.update_num_computed_tokens(num_new_computed_tokens)

    def get_num_uncomputed_tokens(self) -> int:
        num_uncomputed_tokens = 0
        for req in self.get_reqs():
            if not req.is_finished():
                num_uncomputed_tokens += req.data.get_num_uncomputed_tokens()
        return num_uncomputed_tokens

    def num_reqs(self, status: Optional[RequestStatus] = None) -> int:
        # 获取任务中的Request数量
        if status is None:
            return len(self.reqs_dict)

        return len(self.get_reqs(status))

    def num_unfinished_reqs(self) -> int:
        return len(self.get_unfinished_reqs())

    def num_finished_reqs(self) -> int:
        return len(self.get_finished_reqs())

    def find(self, req_id: int) -> Request:
        if req_id not in self.reqs_dict:
            raise ValueError(f"Request {req_id} not found.")
        return self.reqs_dict[req_id]

    def add(self, req: Request) -> None:
        if req.req_id in self.reqs_dict:
            raise ValueError(f"Request {req.req_id} already exists.")
        self.reqs_dict[req.req_id] = req

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.reqs_dict:
            raise ValueError(f"Request {seq_id} not found.")
        del self.reqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(req.is_finished() for req in self.get_reqs())

    def is_prefill(self) -> bool:
        # Every Requests should be in the same stage.
        return self.get_reqs()[0].is_prefill()

    def __repr__(self) -> str:
        return (f"Task(request_id={self.task_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.reqs_dict)})")


class TaskMetadata:
    """任务的元数据。用于创建 `AttentionMetadata`。

    参数:
        request_id: 请求的 ID。
        is_prompt: 请求是否处于提示（prompt）阶段。
        req_data: 序列数据。（req_id -> 序列数据）
        sampling_params: 用于生成输出的采样参数。
        block_tables: 块表。（req_id -> 物理块编号列表）
        do_sample: 如果需要采样则为 True。当不需要采样时，例如在预填充（prefill）被分块的情况下，
            当前迭代仅计算预填充的查询标记时，不需要进行采样。
        token_chunk_size: 每次处理的标记数量（每个序列）。如果不需要分块，则为 None。
        lora_request: LoRA 请求。
        computed_block_nums: 已经计算的块编号，用于前缀缓存。
        state: 与此序列组绑定的内部状态。
        multi_modal_data: 多模态数据。
    """


    def __init__(
        self,
        task_id: str,
        is_prompt: bool,
        req_data: Dict[int, RequestData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
        do_sample: bool = True,
        token_chunk_size: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
        computed_block_nums: Optional[List[int]] = None,
        state: Optional[Task_State] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        self.task_id = task_id
        self.is_prompt = is_prompt
        self.req_data = req_data
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
                self._token_chunk_size = list(req_data.values())[0].get_len()
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
    """The model output associated with a task."""

    def __init__(
        self,
        samples: List[RequestOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        # Prompt logprob for each prompt query token.
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"TaskOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


