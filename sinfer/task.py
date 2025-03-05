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
    class Type(enum.Enum):
        IMAGE = enum.auto()

    def __init__(self, type: Type, data: "torch.Tensor"):
        self.type = type
        self.data = data

@dataclass
class Metadata:
    request_complexity: Optional[float] = None  
    resource_requirements: Optional[dict] = None 
    priority: int = 0  
    deadline: Optional[float] = None 
    arrival_time: float = 0.0 
    last_token_time: float = 0.0
    first_scheduled_time: Optional[float] = None 
    first_token_time: Optional[float] = None
    time_in_pool: Optional[float] = None 
    finished_time: Optional[float] = None 
    prompt_length: Optional[int] = None
    predict_length: Optional[int] = None 

@dataclass
class Task_State:
    generator: Optional = None  # type: ignore

class Task:
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
        return next(iter(self.reqs_dict.values())).prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        return next(iter(self.reqs_dict.values())).data.prompt_token_ids

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def get_last_latency(self, now: float) -> Optional[float]:
        if self.is_prefill():
            raise ValueError(
                "task.get_last_latency() should not be called "
                "if the task is in prefill phase.")

        latency = now - self.metadata.last_token_time
        self.metadata.last_token_time = now
        return latency

    def maybe_set_first_token_time(self, time: float) -> None:
        if (self.metadata.first_token_time is None
                and self.get_reqs()[0].get_output_len() == 1):
            self.metadata.first_token_time = time

    def maybe_set_first_scheduled_time(self, time: float) -> None:
        if self.metadata.first_scheduled_time is None:
            self.metadata.first_scheduled_time = time
            self.metadata.time_in_pool = time - self.metadata.arrival_time

    def set_finished_time(self, time: Optional[float]) -> None:
        self.metadata.finished_time = time

    def get_task_running_reqs(self) -> int:
        if self.sampling_params.use_beam_search:
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_reqs():
                return self.sampling_params.best_of
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
        return self.get_reqs()[0].is_prefill()

    def __repr__(self) -> str:
        return (f"Task(request_id={self.task_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.reqs_dict)})")

class TaskMetadata:
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
        return self._token_chunk_size

class TaskOutput:
    def __init__(
        self,
        samples: List[RequestOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"TaskOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


