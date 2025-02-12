import copy
from enum import IntEnum
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pydantic import Field
from typing_extensions import Annotated

_SAMPLING_EPS = 1e-5


class SamplingType(IntEnum):
    GREEDY = 0 # 贪婪采样
    RANDOM = 1
    RANDOM_SEED = 2
    BEAM = 3

# LogitsProcessor 是一个函数,它接收一个先前生成的 token 列表和一个表示下一个 token 的 logits 张量,并返回一个经过修改的 logits 张量用于采样
LogitsProcessor = Callable[[List[int], torch.Tensor], torch.Tensor]


class SamplingParams:
    """
        采样配置
    参数:
        n: 对给定的 prompt 返回的输出序列数量
        best_of: 从 prompt 生成的序列数量。从这些 best_of 序列中,返回前 n 个最优序列。best_of 必须大于或等于 n。
                当 use_beam_search 为 True 时,这被视为束宽度。默认情况下,best_of 等于 n
        presence_penalty: 一个浮点数,根据生成文本中是否出现新 token 进行惩罚。
                值大于 0 时,鼓励模型使用新 token;值小于 0 时,鼓励模型重复使用 token.
        frequency_penalty: 根据生成文本中 token 的出现频率进行惩罚。值大于 0 时,鼓励模型使用新 token;值小于 0 时,鼓励模型重复使用 token。
        repetition_penalty: 根据 token 是否出现在 prompt 和生成文本中进行惩罚。值大于 1 时,鼓励模型使用新 token;值小于 1 时,鼓励模型重复使用 token。
        temperature: 控制采样的随机性。值越低,模型越确定性;值越高,模型越随机。值为 0 表示贪婪采样。
        top_p: 浮点数,控制采样时考虑的累积概率。范围必须在 (0, 1] 内,设置为 1 时表示考虑所有 token。
        top_k: 一个整数,控制采样时考虑的最高概率的 token 数量。设置为 -1 时表示考虑所有 token.
        min_p: 浮点数,表示 token 的概率相对于最可能 token 的最小值。范围为 [0, 1]。设置为 0 时表示禁用。
        seed: 用于生成的随机种子
        use_beam_search: 是否使用 beam search 而非采样
        length_penalty: 浮点数,根据序列长度对生成进行惩罚,用于 beam search
        early_stopping: 控制 beam search 的停止条件。取值包括:
                    True:当有 best_of 个完整候选时立即停止生成。
                    False:使用启发式方法,当几乎不可能找到更好的候选时停止生成。
                    never:只有当无法找到更好的候选时,才停止(标准 beam search 算法)
        stop: 一个字符串列表,当生成的文本中出现这些字符串时停止生成。返回的输出将不包含这些字符串。
        stop_token_ids: 一个 token 列表,当生成的 token 包含这些 ID 时停止生成。返回的输出将包含这些 token,除非它们是特殊 token.
        include_stop_str_in_output: 是否在输出文本中包含 stop 字符串。默认值为 False
        ignore_eos: 是否忽略 EOS token,在生成 EOS token 后继续生成。
        max_tokens: 每个输出序列生成的最大 token 数。
        min_tokens: 在生成 EOS 或 stop_token_ids 之前,每个输出序列必须生成的最小 token 数。
        logprobs: 返回每个输出 token 的 log 概率数量。结果包括 logprobs 个最可能的 token 的 log 概率,
                以及选定的 token 的 log 概率。API 总是返回采样 token 的 log 概率,因此结果中可能有多达 logprobs+1 个元素
        prompt_logprobs: 返回每个 prompt token 的 log 概率数量
        detokenize: 是否将输出进行解码(detokenize)。默认值为 True
        skip_special_tokens: 是否跳过输出中的特殊 token.
        spaces_between_special_tokens: 是否在输出的特殊 token 之间添加空格。默认值为 True。
        logits_processors: 一个函数列表,根据之前生成的 token 修改 logits
        truncate_prompt_tokens: 如果设置为整数 k,将仅使用 prompt 中最后的 k 个 token(即左截断)。默认为 None(即不截断)
    """

    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        seed: Optional[int] = None,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: Union[bool, str] = False,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: Optional[int] = 16,
        min_tokens: int = 0,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        logits_processors: Optional[List[LogitsProcessor]] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
    ) -> None:
        self.n = n
        self.best_of = best_of if best_of is not None else n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        
        if seed == -1:
            self.seed = None
        else:
            self.seed = seed
        self.use_beam_search = use_beam_search
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        if stop is None:
            self.stop = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = list(stop)
        if stop_token_ids is None:
            self.stop_token_ids = []
        else:
            self.stop_token_ids = list(stop_token_ids)
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.logprobs = logprobs
        self.prompt_logprobs = prompt_logprobs
        # NOTE: This parameter is only exposed at the engine level for now.
        # It is not exposed in the OpenAI API server, as the OpenAI API does
        # not support returning only a list of token IDs.
        self.detokenize = detokenize
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.logits_processors = logits_processors
        self.include_stop_str_in_output = include_stop_str_in_output
        self.truncate_prompt_tokens = truncate_prompt_tokens
        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.output_text_buffer_length = 0

        self._verify_args()
        if self.use_beam_search:
            self._verify_beam_search()
        else:
            self._verify_non_beam_search()
            if self.temperature < _SAMPLING_EPS:
                # Zero temperature means greedy sampling.
                self.top_p = 1.0
                self.top_k = -1
                self.min_p = 0.0
                self._verify_greedy_sampling()
        # eos_token_id is added to this by the engine
        self.all_stop_token_ids = set(self.stop_token_ids)

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if self.best_of < self.n:
            raise ValueError(f"best_of must be greater than or equal to n, "
                             f"got n={self.n} and best_of={self.best_of}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in (0, 2], got "
                             f"{self.repetition_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                             f"got {self.top_k}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1], got "
                             f"{self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be greater than or equal to 0, "
                             f"got {self.min_tokens}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {self.logprobs}.")
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError(f"prompt_logprobs must be non-negative, got "
                             f"{self.prompt_logprobs}.")
        if (self.truncate_prompt_tokens is not None
                and self.truncate_prompt_tokens < 1):
            raise ValueError(f"truncate_prompt_tokens must be >= 1, "
                             f"got {self.truncate_prompt_tokens}")
        if any(not stop_str for stop_str in self.stop):
            raise ValueError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise ValueError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop.")

    def _verify_beam_search(self) -> None:
        if self.best_of == 1:
            raise ValueError("best_of must be greater than 1 when using beam "
                             f"search. Got {self.best_of}.")
        if self.temperature > _SAMPLING_EPS:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")
        if self.early_stopping not in [True, False, "never"]:
            raise ValueError(
                f"early_stopping must be True, False, or 'never', "
                f"got {self.early_stopping}.")

    def _verify_non_beam_search(self) -> None:
        if self.early_stopping is not False:
            raise ValueError("early_stopping is not effective and must be "
                             "False when not using beam search.")
        if (self.length_penalty < 1.0 - _SAMPLING_EPS
                or self.length_penalty > 1.0 + _SAMPLING_EPS):
            raise ValueError(
                "length_penalty is not effective and must be the "
                "default value of 1.0 when not using beam search.")

    def _verify_greedy_sampling(self) -> None:
        if self.best_of > 1:
            raise ValueError("best_of must be 1 when using greedy sampling."
                             f"Got {self.best_of}.")

    # 根据 generation_config 中的非默认值更新 SamplingParams 的配置
    def update_from_generation_config(
            self, generation_config: Dict[str, Any]) -> None:
        if (not self.ignore_eos) and (eos_ids :=
                                      generation_config.get("eos_token_id")):
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids] # 包装成列表
            original_stop_token_ids = set(self.stop_token_ids)  # 包装为set，避免重复token id
            original_stop_token_ids.update(eos_ids) # 合并
            self.stop_token_ids = list(original_stop_token_ids)

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.use_beam_search:
            return SamplingType.BEAM
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    def clone(self) -> "SamplingParams":
        """
            深度copy 除了 LogitsProcessor对象 之外的所有
        """
        logit_processor_refs = None if self.logits_processors is None else {
            id(lp): lp
            for lp in self.logits_processors
        }
        return copy.deepcopy(self, memo=logit_processor_refs)

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"best_of={self.best_of}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"seed={self.seed}, "
            f"use_beam_search={self.use_beam_search}, "
            f"length_penalty={self.length_penalty}, "
            f"early_stopping={self.early_stopping}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"truncate_prompt_tokens={self.truncate_prompt_tokens})")
