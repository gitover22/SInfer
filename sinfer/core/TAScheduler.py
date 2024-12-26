import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sinfer.task import Task, TaskMetadata
from sinfer.config import CacheConfig, LoRAConfig, SchedulerConfig
from sinfer.core.interfaces import AllocStatus, SpaceManager
from sinfer.core.factory import Policy, PolicyFactory
from sinfer.logger import init_logger
from sinfer.lora.request import LoRARequest
from sinfer.request import (Request, RequestData, RequestStatus)
from sinfer.utils import merge_dicts

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("SINFER_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500

class PreemptionStrategy(enum.Enum):
    SWAP = enum.auto() # 换出 kv cache 从gpu换出到cpu
    RECOMPUTE = enum.auto() # 重计算

@dataclass
class SchedulingBudget:
    """
        调度预算
        当前，预算是与`request_id`关联的，这意味着它可以忽略来自相同`request_id`的预算更新。
        这是因为在正常调度路径中,会提前更新正在运行RUNNING的num_seqs, 这可能导致在调度运行中的请求时多次更新。
        由于如果仅使用分块预填充调度,这种情况不会发生,因此当分块预填充默认启用时,可以从API中移除该特性。
    """
    token_budget: int
    max_num_seqs: int
    _requeset_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _requeset_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._requeset_ids_num_batched_tokens:
            return

        self._requeset_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._requeset_ids_num_batched_tokens:
            self._requeset_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._requeset_ids_num_curr_seqs:
            return

        self._requeset_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._requeset_ids_num_curr_seqs:
            self._requeset_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class ScheduledTask:
    # 被调度的序列组
    task: Task
    # 下一次迭代要处理的总分块大小(token数量)
    # 解码时为1,对于预填充,与提示token数量相同,但如果预填充是分块的,则可能会小于提示token数量
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    scheduled_tasks: Iterable[ScheduledTask]  # 被调度的序列组
    num_prefill_tasks: int  # prefill阶段的tasks数量
    num_batched_tokens: int  # 所有scheduled tasks的token总数
    # 换入的块. Dict of CPU -> GPU block number.
    blocks_to_swap_in: Dict[int, int]
    # 换出的块. Dict of GPU -> CPU block number.
    blocks_to_swap_out: Dict[int, int]
    # Blocks to copy. Source to a list of dest blocks.
    blocks_to_copy: Dict[int, List[int]]
    # Sequence groups that are going to be ignored.
    ignored_tasks: List[Task]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int

    def __post_init__(self):
        # 换入和换出不能同时发生
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

    def is_empty(self) -> bool:
        return (not self.scheduled_tasks and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        self.scheduled_tasks = sorted(
            self.scheduled_tasks,
            key=lambda g: (g.task.lora_int_id, g.task.task_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.task.lora_request
            for g in self.scheduled_tasks
            if g.task.lora_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """从 running 队列中调度的request
    可能包含预填充（被分块的预填充数据）或解码数据。如果内存不足，它可以被抢占（用于重新计算）或被交换出去
    """

    decode_tasks: List[Task]    # decode阶段的序列组
    prefill_tasks: List[Task]   # prefill阶段的序列组（意味着开启了 chunked prefill）
    preempted: List[Task]            # 被抢占的序列组(重计算)
    swapped_out: List[Task]          # swap out的序列组(换出)

    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_tasks=[],
            prefill_tasks=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out={},
            blocks_to_copy={},
            num_lookahead_slots=0,
        )


@dataclass
class SchedulerSwappedInOutputs:
    """从 swap 队列调度的task

    可能包含prefill (chunked prefill) 和 decodes阶段的序列组.
    """
    # Selected tasks that are going to be swapped in and is in decoding phase.
    decode_tasks: List[Task]
    # Selected tasks that are going to be swapped in and in a prefill
    # phase. （意味着chunked prefill开启）
    prefill_tasks: List[Task]
    blocks_to_swap_in: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible tasks.
    infeasible_tasks: List[Task]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_tasks=[],
            prefill_tasks=[],
            blocks_to_swap_in={},
            blocks_to_copy={},
            num_lookahead_slots=0,
            infeasible_tasks=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """从waiting队列调度的序列组
    可能包含新的预填充请求或被抢占的请求(抢占策略为重计算),这些请求需要从头重新计算
    """
    # Selected tasks for prefill.
    tasks: List[Task]
    # Ignored tasks
    ignored_tasks: List[Task]
    num_lookahead_slots: int
    # 创建一个对象
    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            tasks=[],
            ignored_tasks=[],
            num_lookahead_slots=0,
        )

class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # LoRA配置
        self.lora_config = lora_config

        if self.scheduler_config.chunked_prefill_enabled:
            self.prompt_limit = self.scheduler_config.max_model_len
        else:
            self.prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens)
        # 任务池
        self.RunningPool: Deque[Task] = deque()  # 存储运行态的任务（存储处于decode阶段的任务和chunked prefill）
        self.ReadyPool: Deque[Task] = deque()  # 存储就绪态任务（存储新来的处于prefill阶段的任务和重计算的任务）waiting队列中的task只包含一个req，即是原始的prompt
        self.BlockedPool: Deque[Task] = deque() # 存储阻塞态的任务，包含被换出的处于decode阶段的任务，或者chunked prefill开启时存储prefill阶段的任务

        BlockSpaceManagerImpl = SpaceManager.get_block_space_manager()
        # 创建物理块管理器
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_mlu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)


        self.prev_time = 0.0   # 每执行1次推理阶段前，调度器都要做一次调度，这个变量存放的就是上次调度发起的时间点
        self.prev_prompt = False  # 上一次调度时，调度器有从waiting队列中取出task做推理，即为True，否则为False
        self.last_prompt_latency = 0.0 # 当前调度时刻（now） - 最后一次有从waiting队列中取数做推理的那个调度时刻

        # 以下字段仅用于测试。它被用来注入人工抢占
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        
        # 分析调度器性能
        logger.info(f"SInfer TAScheduler profiling...")
        
        self.df = pd.DataFrame(
            data={
                'waiting': [], 'running': [], 'swapped': [], 'finished': [],
                'batch_utils': [], 'block_utils': [], 'preempt_ratio': []
            },
            dtype=np.float32
        )
        self.sched_step = 0
        self.running_seqs = 0
        self.waiting_seqs = 0
        self.swapped_seqs = 0
        self.finished_seqs = 0
        self.total_seqs = 0
        self.running_to_waiting_seqs = 0
        self.batch_utils = 0
        self.block_utils = 0
        self.preempt_ratio = 0

        self.finished_tasks = []

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_req(self) -> int:
        """The number of new tokens."""
        return 1

    def add_task(self, task: Task) -> None:
        # 往waiting队列添加序列组
        self.ReadyPool.append(task)

    def abort_task(self, task_id: Union[str, Iterable[str]]) -> None:
        """
        中止给定 ID 的序列组
        检查给定 ID 的序列组是否存在于任何状态队列中。
        如果存在，从状态队列中移除该序列组。如果序列组中的任何序列尚未完成，则将该序列的状态设置为 FINISHED_ABORTED 并释放它

        参数：
            task_id: 要中止的序列组的 ID
        """
        if isinstance(task_id, str):
            task_id = (task_id, )
        task_ids = set(task_id)
        for state_queue in [self.ReadyPool, self.RunningPool, self.BlockedPool]:
            aborted_tasks: List[Task] = []
            for task in state_queue:
                if not task_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if task.task_id in task_ids:
                    # Appending aborted group into pending list.
                    aborted_tasks.append(task)
                    task_ids.remove(task.task_id)
            for aborted_task in aborted_tasks:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_task)
                for seq in aborted_task.get_reqs():
                    if seq.is_finished():
                        continue
                    seq.status = RequestStatus.FINISHED_ABORTED
                    self.free_req(seq)

    def has_unfinished_tasks(self) -> bool:
        '''是否还有没完成的序列'''
        return len(self.ReadyPool) != 0 or len(self.RunningPool) != 0 or len(
            self.BlockedPool) != 0

    def get_num_unfinished_tasks(self) -> int:
        '''获取未完成的序列组'''
        return len(self.ReadyPool) + len(self.RunningPool) + len(self.BlockedPool)

    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """调度 running 队列中的SG

        Running queue should include decode and chunked prefill requests.

        Args:
            running_queue: The queue that contains running requests (i.e.,
                decodes). The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            policy: The sorting policy to sort running_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            A tuple of remaining running queue (should be always 0) after
            scheduling and SchedulerRunningOutputs.
        """
        init_queue = running_queue
        # 模型执行前需要swap和copy的块
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        decode_seq_groups: List[ScheduledTask] = []
        prefill_seq_groups: List[ScheduledTask] = []
        preempted: List[Task] = []
        swapped_out: List[Task] = []

        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)
        while running_queue:
            task = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                task, RequestStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            while not self._can_append_slots(task):
                budget.subtract_num_batched_tokens(task.task_id,
                                                   num_running_tokens)
                num_running_seqs = task.get_max_num_running_seqs()
                budget.subtract_num_seqs(task.task_id,
                                         num_running_seqs)
                if curr_loras is not None and task.lora_int_id > 0:
                    curr_loras.remove(task.lora_int_id)

                if running_queue:
                    # Preempt the lowest-priority tasks
                    victim_seq_group = running_queue.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionStrategy.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                else:
                    # No other tasks can be preempted. Preempt the current task.
                    preempted_mode = self._preempt(task,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionStrategy.RECOMPUTE:
                        preempted.append(task)
                    else:
                        swapped_out.append(task)
                    break
            else:
                self._append_slots(task, blocks_to_copy)
                is_prefill = task.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledTask(
                            task=task,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledTask(task=task,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(task.task_id,
                                              num_running_tokens)
                # 注意，get_max_num_running_seqs开销较大。在默认的调度策略中，当 enable_chunking为 False 时，
                # num_seqs会在运行此方法之前更新，因此我们在这里不需要再次更新它。
                if enable_chunking:
                    num_running_seqs = task.get_max_num_running_seqs()
                    budget.add_num_seqs(task.task_id, num_running_seqs)
                if curr_loras is not None and task.lora_int_id > 0:
                    curr_loras.add(task.lora_int_id)
        # add for scheduler profiling
        # run_reqs = len(init_queue)
        # sched_reqs = run_reqs - len(running_queue)
        sched_to_preempt = [s.task_id for s in preempted]
        # sched_to_swap_out = [s.request_id for s in swapped_out]
        self.running_to_waiting_seqs += len(sched_to_preempt)
        return running_queue, SchedulerRunningOutputs(
            decode_tasks=decode_seq_groups,
            prefill_tasks=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

    def _schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerSwappedInOutputs]:
        """调度被swap out的序列组

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            swapped_queue: The queue that contains swapped out requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            policy: The sorting policy to sort swapped_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining swapped_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        # 模型执行前需要swap和copy的块
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        decode_seq_groups: List[ScheduledTask] = []
        prefill_seq_groups: List[ScheduledTask] = []

        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)
        infeasible_tasks: List[Task] = []

        leftover_swapped: Deque[Task] = deque()
        while swapped_queue:
            task = swapped_queue[0]

            alloc_status = self.block_manager.can_swap_in(task)
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the task %s because there's not enough kv "
                    "cache blocks to run the entire request.",
                    task.task_id)
                for req in task.get_reqs():
                    req.status = RequestStatus.FINISHED_IGNORED
                infeasible_tasks.append(task)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = task.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so we ignore
                    leftover_swapped.appendleft(task)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = task.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(task,
                                                      RequestStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(task, blocks_to_swap_in)
            self._append_slots(task, blocks_to_copy)
            is_prefill = task.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledTask(task,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledTask(task, token_chunk_size=1))
            budget.add_num_batched_tokens(task.task_id, num_new_tokens)
            budget.add_num_seqs(task.task_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)
        # add for scheduler profiling
        # swap_reqs = len(swapped_queue)
        # sched_reqs = swap_reqs - len(swapped_queue)
        # sched_to_swap_in = [s.task.task_id for s in decode_seq_groups]
        return swapped_queue, SchedulerSwappedInOutputs(
            decode_tasks=decode_seq_groups,
            prefill_tasks=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_tasks=infeasible_tasks,
        )

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """调度prefill 阶段的序列组

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        init_queue = waiting_queue
        ignored_tasks: List[Task] = []
        tasks: List[Task] = []
        # 不对等待队列进行排序，因为假设它已经是有序的
        # 复制该队列以确保输入队列不会被修改
        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_tasks: Deque[Task] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            task = waiting_queue[0]

            waiting_seqs = task.get_reqs(status=RequestStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting task should have only one prompt request.")
            num_new_tokens = self._get_num_new_tokens(task,
                                                      RequestStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            if num_new_tokens > self.prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens,
                    self.prompt_limit)
                for seq in waiting_seqs:
                    seq.status = RequestStatus.FINISHED_IGNORED
                ignored_tasks.append(task)
                waiting_queue.popleft()
                continue

            # 判断能不能给这个SG分配
            can_allocate = self.block_manager.can_allocate(task)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = RequestStatus.FINISHED_IGNORED
                ignored_tasks.append(task)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = task.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this task for now.
                    leftover_waiting_tasks.appendleft(task)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = task.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # 可以调度这个序列
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(task) # 实际分配
            tasks.append(
                ScheduledTask(task=task,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(task.task_id, num_new_tokens)
            budget.add_num_seqs(task.task_id, num_new_seqs)

        # 将无法被调度的请求重新加入队列中
        waiting_queue.extendleft(leftover_waiting_tasks)
        if len(tasks) > 0:
            self.prev_prompt = True

        # add for scheduler profiling
        # wait_reqs = len(init_queue)
        # sched_reqs = wait_reqs - len(waiting_queue)
        # sched_to_run = [s.task.task_id for s in prefills.seq_groups]
        # sched_to_ingore = [s.task.task_id for s in prefills.ignored_seq_groups]

        return waiting_queue, SchedulerPrefillOutputs(
            tasks=tasks,
            ignored_tasks=ignored_tasks,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _schedule_default(self) -> SchedulerOutputs:
        """
        running队列和blocked队列全是decode阶段(no chunked prefill)
        任务池优先级: ReadyPool > RunningPool > BlockedPool
        此策略旨在优化吞吐量。首先，它将尽可能多的预填充请求进行批处理。
        然后它调度解码请求。如果MLU内存出现压力，解码请求可以被交换或抢占
        """ 
        # 调度预算：主要关注tokens总数和seq总数
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens, # 最大：2048
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # 确保在调度prefill请求之前包含当前运行中的序列数，  
        # 以避免预填充的调度超出 `max_num_seqs` 的限制。
        for task in self.RunningPool:
            budget.add_num_seqs(task.task_id,
                                task.get_max_num_running_seqs())
        curr_loras = set(
            task.lora_int_id
            for task in self.RunningPool) if self.lora_enabled else None # 启用了LoRA

        remaining_waiting, prefills = (self.ReadyPool,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.RunningPool, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.BlockedPool, SchedulerSwappedInOutputs.create_empty())

        # 如果swapped队列为空，调度waiting队列
        if not self.BlockedPool:
            remaining_waiting, prefills = self._schedule_prefills(
                self.ReadyPool, budget, curr_loras, enable_chunking=False)

        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        # fcfs_policy = PolicyFactory.get_policy(policy_name="tas")
        # 如果已经调度了预填充请求，则不要再调度解码请求
        # 如果 _schedule_prefills 未启用分块模式（chunking），
        # 那么self.running 中仅包含解码请求，而不包含分块的预填充请求
        if len(prefills.tasks) == 0: # 没有调度prefill
            remaining_running, running_scheduled = self._schedule_running(
                self.RunningPool,
                budget,
                curr_loras,
                fcfs_policy,
                enable_chunking=False)

            # 如果有任何序列组被抢占或换出，则不要swap in任何序列组，  
            # 因为这意味着没有可用的插槽用于新的运行请求（running_scheduled.preempted存的是重计算的SG,running_scheduled.swapped_out存的是换出的SG）
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                remaining_swapped, swapped_in = self._schedule_swapped(
                    self.BlockedPool, budget, curr_loras, fcfs_policy)
        # 确保预算内的token数和序列数没有超出限制
        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update
        self.ReadyPool = remaining_waiting
        self.ReadyPool.extendleft(running_scheduled.preempted) # 重计算的SG加入到waiting队列最左侧
        # 
        self.RunningPool = remaining_running
        self.RunningPool.extend([s.task for s in prefills.tasks])
        self.RunningPool.extend(
            [s.task for s in running_scheduled.decode_tasks])
        self.RunningPool.extend(
            [s.task for s in swapped_in.decode_tasks])
        # 
        self.BlockedPool = remaining_swapped
        self.BlockedPool.extend(running_scheduled.swapped_out) # 换出的SG加入到swapped队列

        # 没开启chunked prefill时，running队列和swapped队列都不应该包含prefill阶段的SG
        assert len(running_scheduled.prefill_tasks) == 0
        assert len(swapped_in.prefill_tasks) == 0
        # add for scheduler profiling
        self.sched_step += 1
        # 包装一下调度结果并返回
        return SchedulerOutputs(
            scheduled_tasks=(prefills.tasks +
                                  running_scheduled.decode_tasks +
                                  swapped_in.decode_tasks),
            num_prefill_tasks=len(prefills.tasks),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=merge_dicts(running_scheduled.blocks_to_copy,
                                       swapped_in.blocks_to_copy),
            ignored_tasks=prefills.ignored_tasks +
            swapped_in.infeasible_tasks,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.RunningPool),
        )

    def _schedule_chunked_prefill(self):
        """使用chunked prefill时的调度
        
        分块预填充(Chunked Prefill)允许将预填充请求拆分成块,与解码请求一起批量处理:
        1. 调度尽可能多的解码请求
        2. 调度未完成的分块预填充请求
        3. 调度被交换(swapped)的请求
        4. 调度新的预填充请求

        该策略能够维持较高的GPU利用率, 因为它可以将预填充请求与解码请求合并到同一批次中
        同时,它还能提高每个token的延迟性能( inter-token latency), 因为解码请求无需被预填充请求阻塞
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.ReadyPool,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.RunningPool, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.BlockedPool, SchedulerSwappedInOutputs.create_empty())

        # Decoding should be always scheduled first by fcfs.
        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        # fcfs_policy = PolicyFactory.get_policy(policy_name="tas")
        remaining_running, running_scheduled = self._schedule_running(
            self.RunningPool,
            budget,
            curr_loras,
            fcfs_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            remaining_swapped, swapped_in = self._schedule_swapped(
                self.BlockedPool, budget, curr_loras, fcfs_policy)

        # Schedule new prefills.
        remaining_waiting, prefills = self._schedule_prefills(
            self.ReadyPool, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.ReadyPool = remaining_waiting
        self.ReadyPool.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.RunningPool = remaining_running
        self.RunningPool.extend([s.task for s in prefills.tasks])
        self.RunningPool.extend(
            [s.task for s in running_scheduled.decode_tasks])
        self.RunningPool.extend(
            [s.task for s in running_scheduled.prefill_tasks])
        self.RunningPool.extend(
            [s.task for s in swapped_in.decode_tasks])
        self.RunningPool.extend(
            [s.task for s in swapped_in.prefill_tasks])
        # Update swapped requests.
        self.BlockedPool = remaining_swapped
        self.BlockedPool.extend(running_scheduled.swapped_out)
        return SchedulerOutputs(
            scheduled_tasks=(prefills.tasks +
                                  running_scheduled.prefill_tasks +
                                  swapped_in.prefill_tasks +
                                  running_scheduled.decode_tasks +
                                  swapped_in.decode_tasks),
            num_prefill_tasks=(len(prefills.tasks) +
                                len(swapped_in.prefill_tasks) +
                                len(running_scheduled.prefill_tasks)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=merge_dicts(running_scheduled.blocks_to_copy,
                                       swapped_in.blocks_to_copy),
            ignored_tasks=prefills.ignored_tasks,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.RunningPool),
        )

    def _schedule(self) -> SchedulerOutputs:
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default() # 默认

    def _can_append_slots(self, task: Task) -> bool:
        """
        确定KV缓存中是否有足够的空间以继续生成序列组
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        # Appending slots only occurs in decoding.
        is_prefill = False

        return self.block_manager.can_append_slots(
            task=task,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )

    def schedule(self) -> Tuple[List[TaskMetadata], SchedulerOutputs]:
        '''
        调度器的调度入口，会封装调度输出
        '''
        scheduler_outputs = self._schedule() # 获取调度输出
        now = time.time()

        seq_group_metadata_list: List[TaskMetadata] = []
        # 依次遍历每一个序列组
        for i, scheduled_task in enumerate(
                scheduler_outputs.scheduled_tasks):
            task = scheduled_task.task
            token_chunk_size = scheduled_task.token_chunk_size
            task.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            req_data: Dict[int, RequestData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}
            # 依次遍历SG中的所有seq
            for req in task.get_reqs(status=RequestStatus.RUNNING):
                req_id = req.req_id
                req_data[req_id] = req.data
                block_tables[req_id] = self.block_manager.get_block_table(req)
                self.block_manager.access_all_blocks_in_seq(req, now)

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    task.get_reqs(status=RequestStatus.RUNNING)))

            do_sample = True
            if task.is_prefill():
                seqs = task.get_reqs()
                assert len(seqs) == 1  # Prefill 阶段只有一个seq
                # chunked prefill不需要进行采样。  
                if (token_chunk_size + seqs[0].data.get_num_computed_tokens() <
                        seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            is_prompt = task.is_prefill()
            seq_group_metadata = TaskMetadata(
                task_id=task.task_id,
                is_prompt=is_prompt,
                req_data=req_data,
                sampling_params=task.sampling_params,
                block_tables=block_tables,
                do_sample=do_sample,
                token_chunk_size=token_chunk_size,
                lora_request=task.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=task.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=task.multi_modal_data
                if scheduler_outputs.num_prefill_tasks > 0 else None,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # 假设批次中的所有块都会在下次调度调用之前完成计算
        # 这是因为模型执行中的失败会导致 SInfer 实例崩溃，而不会进行重试
        for scheduled_task in scheduler_outputs.scheduled_tasks:
            self.block_manager.mark_blocks_as_computed(
                scheduled_task.task)

        return seq_group_metadata_list, scheduler_outputs

    def fork_req(self, parent_req: Request, child_req: Request) -> None:
        self.block_manager.fork(parent_req, child_req)

    def free_req(self, req: Request) -> None:
        """释放一个序列"""
        self.block_manager.free(req)

    # 释放已经完成的序列组
    def free_finished_tasks(self) -> None:
        finished_tasks_ = [task for task in self.RunningPool
                                        if task.is_finished()]

        self.RunningPool = deque(task for task in self.RunningPool
                                if not task.is_finished())

        self.running_seqs = len(self.RunningPool)
        self.waiting_seqs = len(self.ReadyPool)
        self.swapped_seqs = len(self.BlockedPool)
        self.finished_seqs += len(finished_tasks_)
        self.finished_tasks += finished_tasks_

        total_seqs_ = self.running_seqs + self.waiting_seqs + self.swapped_seqs + self.finished_seqs
        if total_seqs_ > self.total_seqs:
            self.total_seqs = total_seqs_

        self.batch_utils = self.running_seqs / self.scheduler_config.max_num_seqs
        self.block_utils = (self.block_manager.num_total_gpu_blocks -
                            self.block_manager.get_num_free_gpu_blocks()) / self.block_manager.num_total_gpu_blocks
        self.preempt_ratio = self.running_to_waiting_seqs / self.total_seqs

        df_ = pd.DataFrame(
            [[self.waiting_seqs, self.running_seqs, self.swapped_seqs,
            len(finished_tasks_), self.batch_utils, self.block_utils, self.preempt_ratio]],
            columns=['waiting', 'running', 'swapped',
                    'finished','batch_utils', 'block_utils', 'preempt_ratio'],
            index=[str(self.sched_step)])
        self.df = pd.concat([self.df, df_])

    def _allocate_and_set_running(self, task: Task) -> None:
        self.block_manager.allocate(task)
        for req in task.get_reqs(status=RequestStatus.WAITING):
            req.status = RequestStatus.RUNNING

    def _append_slots(
        self,
        task: Task,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        """Appends new slots to the Reqs in the given task.

        Args:
            task: containing the Requests to append slots to.
            blocks_to_copy (Dict[int, List[int]]): A dictionary mapping source
                block indices to lists of destination block indices. This
                dictionary is updated with the new source and destination block
                indices for the appended slots.
        """
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False)
        # 遍历这个seq_group中状态为running的所有seq
        for req in task.get_reqs(status=RequestStatus.RUNNING):
            cows = self.block_manager.append_slots(req, num_lookahead_slots)

            for src, dests in cows.items():
                if src not in blocks_to_copy:
                    blocks_to_copy[src] = []
                blocks_to_copy[src].extend(dests)

    def _preempt(
        self,
        task: Task,
        blocks_to_swap_out: Dict[int, int],
        preemption_strategy: Optional[PreemptionStrategy] = None,
    ) -> PreemptionStrategy:
        # 如果未指定抢占策略，将根据以下规则确定策略：
        # 默认使用重计算，因为其开销比交换模式更低。然而，当序列组包含多个序列（例如，束搜索）时，使用交换模式。
        # 缺陷：被交换的序列优先于等待的序列，所以包含多个序列的序列组隐式地优先于仅包含单个序列的序列组。
        # 为包含多个序列的序列组支持重计算。这需要更复杂的CUDA内核支持
        if preemption_strategy is None:
            if task.get_max_num_running_seqs() == 1:
                preemption_strategy = PreemptionStrategy.RECOMPUTE
            else:
                preemption_strategy = PreemptionStrategy.SWAP
        if preemption_strategy == PreemptionStrategy.RECOMPUTE:
            self._preempt_by_recompute(task)
        elif preemption_strategy == PreemptionStrategy.SWAP:
            self._preempt_by_swap(task, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_strategy

    def _preempt_by_recompute(
        self,
        task: Task,
    ) -> None:
        reqs = task.get_reqs(status=RequestStatus.RUNNING)  # 获取这个seq_group下正在running的所有seqs
        assert len(reqs) == 1 # 重计算时SG中的seq数量为1才正确
        for req in reqs:
            req.status = RequestStatus.WAITING 
            # 释放这条seq对应的物理块
            # 即将对应物理块的引用-1，如果此时引用数量为0，说明对应物理块完全自由了，需要再将其放入自由物理块列表中
            self.free_req(req)
            req.reset_state_for_recompute()

    def _preempt_by_swap(
        self,
        task: Task,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        # 释放该seq_group下所有seq的物理块，并为其分配对应的cpu物理块，
        # 将seq的状态从running改成swapped
        self._swap_out(task, blocks_to_swap_out)

    def _swap_in(
        self,
        task: Task,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(task)
        blocks_to_swap_in.update(mapping)
        for req in task.get_reqs(status=RequestStatus.SWAPPED):
            req.status = RequestStatus.RUNNING

    def _swap_out(
        self,
        task: Task,   # 需要被swap到cpu上的seq_group
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(task):
            # 检查是否可以将当前seq_group对应的物理块swap到cpu上
            # 可以的条件：当前seq_group占用的gpu物理块数量 <= cpu上可用的物理块数量
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        # 释放该seq_group下所有seq的gpu物理块，并为其创建对应的cpu块
        # mapping：{gpu物理块id：cpu物理块id}
        mapping = self.block_manager.swap_out(task)
        blocks_to_swap_out.update(mapping)
        for req in task.get_reqs(status=RequestStatus.RUNNING):
            req.status = RequestStatus.SWAPPED # 修改该seq_group下所有seq的状态：从running改成swapped

    def _passed_delay(self, now: float) -> bool:
        """
        判断当下是否可以从waiting队列中调度新请求
        这个函数确保了在调度过程中不会频繁地处理新来的seq_group
        
        Args:
            now: 当前调度时间点
        """
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        # 用当前调度时间更新prev_time
        # 由于目前还不知道本次是否会从waiting队列中调度，因此prev_prompt先设为False
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        # delay_factor：用户配置的，用于调整调度间隔阈值的因子。大于0则意味着用户想开启阈值判断
        if self.scheduler_config.delay_factor > 0 and self.ReadyPool:
            # 计算在waiting队列中，最早到达的seq_group的到达时间
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.ReadyPool])
            # 实际等待的时间 > 应该等待时间
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.RunningPool)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool) -> int:
        """
        每个序列在每个步骤中分配的额外插槽数量, 用于存储除已知token ID以外的内容
        推测性解码使用这些插槽来存储可能被接受或可能不被接受的token的KV激活
        推测性解码目前尚不支持预填充, 因此我们不会为预填充执行前瞻性分配
        """
        if is_prefill:
            return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_tokens(self, task: Task,
                            status: RequestStatus, enable_chunking: bool,
                            budget: SchedulingBudget) -> int:
        """
            获取处于指定 status 下的序列组需要计算的下一个新 token
            如果`enable_chunking`为True, API可以根据`budget`将需要计算的token数量分块
            如果一个序列组包含多个序列（例如, 在运行束搜索时）, 则表明它处于解码阶段，因此不会进行分块
            如果由于token预算限制无法计算新token, 则返回0。
        """
        num_new_tokens = 0
        reqs = task.get_reqs(status=status)
        for req in reqs:
            num_new_tokens += req.get_num_new_tokens()
        assert num_new_tokens > 0
        # 如果正在运行的请求无法容纳，则进行分块
        # 如果序列数大于1，则表示正在解码阶段执行束搜索。在这种情况下，不进行分块。
        if enable_chunking and len(reqs) == 1:
            num_new_tokens = min(num_new_tokens,
                                 budget.remaining_token_budget())
        return num_new_tokens
    
    def summary_finished_tasks(self,tasks: List[Task]):
        df = pd.DataFrame(
        data={
            'ttft/s': [], 'time_in_queue/s': [], 'context_latency/s': [], 'decoder_latency/s': []
        },
        dtype=np.float32
        )
        for task in tasks:
            ttft = task.metrics.first_token_time - task.metrics.arrival_time
            time_in_queue = task.metrics.time_in_queue
            context_latency = task.metrics.first_token_time - task.metrics.first_scheduled_time
            decoder_latency = task.metrics.finished_time - task.metrics.first_token_time
            decoder_token_num = task.get_reqs()[0].get_output_len() - 1
            per_token_latency = decoder_latency if decoder_token_num == 0 \
                                                else decoder_latency / decoder_token_num
            df_ = pd.DataFrame(
                [[ttft, time_in_queue, context_latency, decoder_latency, per_token_latency, decoder_token_num]],
                columns=['ttft/s', 'time_in_queue/s', 'context_latency/s', 'decoder_latency/s', 'per_token_latency/s', 'decoder_tokens'],
                index=[str(task.task_id)]
            )
            df = pd.concat([df, df_])
        sum_, max_, mean_, min_, p99_ = df.sum(), df.max(), df.mean(), df.min(), df.quantile(0.99)
        df.loc['Sum'] = sum_
        df.loc['Max'] = max_
        df.loc['Mean'] = mean_
        df.loc['Min'] = min_
        df.loc['P99'] = p99_
        return df

    def __del__(self):
        logger.info(f"SInfer TAScheduler profiling saving...")
        figure = self.df.plot(y=['waiting', 'running', 'swapped', 'finished']).get_figure()
        figure.savefig("sinfer_queue_view.jpg", dpi=300)
        plt.close(figure)
        figure = self.df.plot(y=['batch_utils', 'block_utils', 'preempt_ratio']).get_figure()
        figure.savefig("sinfer_utilization_view.jpg", dpi=300)
        plt.close(figure)

        time_df = self.summary_finished_tasks(self.finished_tasks)

        max_, mean_, min_ = self.df.max(), self.df.mean(), self.df.min()
        self.df.loc["Max"] = max_
        self.df.loc["Mean"] = mean_
        self.df.loc["Min"] = min_
        with pd.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.max_colwidth', None,
                            'display.float_format', '{:^6,.2f}'.format,
                            'expand_frame_repr', False):
            logger.info(self.df.loc[["Max", "Mean", "Min"]])
            logger.info(time_df.loc[["Sum", "Max", "Mean", "Min", "P99"]])
        self.df.astype(str).to_csv("sinfer_scheduler_view.csv", mode="w")
        time_df.astype(str).to_csv("sinfer_time_view.csv", mode="w")
