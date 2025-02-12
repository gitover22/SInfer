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
from sinfer.scheduler.interfaces import AllocStatus, SpaceManager
from sinfer.scheduler.factory import Profiler, Profilers
from sinfer.logger import init_logger
from sinfer.lora.request import LoRARequest
from sinfer.request import (Request, RequestData, RequestStatus)
from sinfer.utils import merge_dicts

logger = init_logger(__name__)

class PreemptionStrategy(enum.Enum):
    BLOCK = enum.auto()
    RECOMPUTE = enum.auto()

@dataclass
class SchedulingBudget:
    """
        调度预算
        当前，预算是与`request_id`关联的，这意味着它可以忽略来自相同`request_id`的预算更新。
        这是因为在正常调度路径中,会提前更新正在运行RUNNING的num_reqs, 这可能导致在调度运行中的请求时多次更新。
        仅使用分块预填充调度,这种情况不会发生,因此当分块预填充默认启用时,可以从API中移除该特性。
    """
    token_budget: int
    max_num_reqs: int
    _requeset_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _requeset_ids_num_curr_reqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _curr_reqs_num: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_reqs: int):
        assert num_new_tokens != 0
        assert num_new_reqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_reqs + num_new_reqs <= self.max_num_reqs)

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

    def add_num_reqs(self, req_id: str, curr_reqs_num: int):
        if req_id in self._requeset_ids_num_curr_reqs:
            return

        self._requeset_ids_num_curr_reqs.add(req_id)
        self._curr_reqs_num += curr_reqs_num

    def subtract_num_reqs(self, req_id: str, curr_reqs_num: int):
        if req_id in self._requeset_ids_num_curr_reqs:
            self._requeset_ids_num_curr_reqs.remove(req_id)
            self._curr_reqs_num -= curr_reqs_num

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_reqs(self):
        return self._curr_reqs_num


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
class SchedulerBlockInOutputs:
    """从阻塞池调度的task

    可能包含prefill (chunked prefill) 和 decodes阶段的序列组.
    """
    # Selected tasks that are going to be BlockIn and is in decoding phase.
    decode_tasks: List[Task]
    # Selected tasks that are going to be BlockIn and in a prefill phase(cp open
    prefill_tasks: List[Task]
    blocks_to_blockIn: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible tasks.
    infeasible_tasks: List[Task]

    @classmethod
    def create_empty(cls) -> "SchedulerBlockInOutputs":
        return SchedulerBlockInOutputs(
            decode_tasks=[],
            prefill_tasks=[],
            blocks_to_blockIn={},
            blocks_to_copy={},
            num_lookahead_slots=0,
            infeasible_tasks=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    # 从就绪池调度的任务：新的预填充请求或被重计算的请求
    tasks: List[Task]
    ignored_tasks: List[Task]
    _lookahead_slots_num: int
    # 创建一个对象
    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            tasks=[],
            ignored_tasks=[],
            _lookahead_slots_num=0,
        )

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("SINFER_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500

class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        
        self.lora_config = lora_config

        if self.scheduler_config.chunked_prefill_enabled:
            self.prompt_limit = self.scheduler_config.max_model_len
        else:
            self.prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens)
            
        self._latency_lastFromReadyPool = 0.0 # now - 上一次有从就绪队列中取任务
        self._last_schedule_formReadyPool = False
        self._last_schedule_time = 0.0 
        
        self.RunningPool: Deque[Task] = deque()  # decode + chunked prefill
        self.ReadyPool: Deque[Task] = deque()  # new prefill + recompute prefill
        self.BlockedPool: Deque[Task] = deque() # swapped decode + chunked prefill

        BlockSpaceManager = SpaceManager.get_block_space_manager()
        
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_mlu_blocks=self.cache_config.num_mlu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # for test 它被用来注入人工抢占
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        
        logger.info(f"SInfer TAScheduler profiling...")
        self.df = pd.DataFrame(
            data={
                'running': [],'ready': [], 'finished': [],'batch_utils': [], 'block_utils': []
            },
            dtype=np.float32
        )
        self.ready_tasks = 0
        self.running_tasks = 0
        self.blocked_tasks = 0
        self.finished_tasks = 0
        self.finished_tasks_list = []
        self.total_tasks = 0
        self.schedule_step = 0
        self.from_running_to_ready_tasks = 0
        self.batch_utils = 0
        self.block_utils = 0
        self.preempt_ratio = 0


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

    def _schedule_running_task(
        self,
        running_pool: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Profiler,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """调度运行池中的task
        Args:
            running_pool: contains running requests (i.e.,
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
        init_queue = running_pool
        # 模型执行前需要blockout和copy的块
        blocks_to_BlockOut: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        decode_tasks: List[ScheduledTask] = []
        prefill_tasks: List[ScheduledTask] = []
        preempted: List[Task] = []
        swapped_out: List[Task] = []

        now = time.time()
        running_pool = policy.sort_by_score(now, running_pool)
        while running_pool:
            task = running_pool[0]
            num_running_tokens = self._get_num_new_tokens(
                task, RequestStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_pool.popleft()
            while not self._MLU_memory_sufficient(task):
                budget.subtract_num_batched_tokens(task.task_id,
                                                   num_running_tokens)
                num_running_seqs = task.get_max_num_running_seqs()
                budget.subtract_num_reqs(task.task_id,
                                         num_running_seqs)
                if curr_loras is not None and task.lora_int_id > 0:
                    curr_loras.remove(task.lora_int_id)

                if running_pool:
                    # Preempt the lowest-priority tasks
                    victim_seq_group = running_pool.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_BlockOut)
                    if preempted_mode == PreemptionStrategy.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                else:
                    # No other tasks can be preempted. Preempt the current task.
                    preempted_mode = self._preempt(task,
                                                   blocks_to_BlockOut)
                    if preempted_mode == PreemptionStrategy.RECOMPUTE:
                        preempted.append(task)
                    else:
                        swapped_out.append(task)
                    break
            else:
                self._append_slots(task, blocks_to_copy)
                is_prefill = task.is_prefill()
                if is_prefill:
                    prefill_tasks.append(
                        ScheduledTask(
                            task=task,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_tasks.append(
                        ScheduledTask(task=task,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(task.task_id,
                                              num_running_tokens)
                # 注意，get_max_num_running_seqs开销较大。在默认的调度策略中，当 enable_chunking为 False 时，
                # num_seqs会在运行此方法之前更新，因此我们在这里不需要再次更新它。
                if enable_chunking:
                    num_running_seqs = task.get_max_num_running_seqs()
                    budget.add_num_reqs(task.task_id, num_running_seqs)
                if curr_loras is not None and task.lora_int_id > 0:
                    curr_loras.add(task.lora_int_id)
        # add for scheduler profiling
        # run_reqs = len(init_queue)
        # sched_reqs = run_reqs - len(running_queue)
        sched_to_preempt = [s.task_id for s in preempted]
        # sched_to_swap_out = [s.request_id for s in swapped_out]
        self.from_running_to_ready_tasks += len(sched_to_preempt)
        return running_pool, SchedulerRunningOutputs(
            decode_tasks=decode_tasks,
            prefill_tasks=prefill_tasks,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_BlockOut,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

    def _schedule_blocked_task(
        self,
        blocked_pool: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Profiler,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerBlockInOutputs]:
        """调度被swap out的序列组

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            blocked_pool: contains swapped out requests.
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
        blocks_to_BlockIn: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        decode_tasks: List[ScheduledTask] = []
        prefill_tasks: List[ScheduledTask] = []

        now = time.time()
        blocked_pool = policy.sort_by_score(now, blocked_pool)
        infeasible_tasks: List[Task] = []

        leftover_swapped: Deque[Task] = deque()
        while blocked_pool:
            task = blocked_pool[0]

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
                blocked_pool.popleft()
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
                    blocked_pool.popleft()
                    continue

            # The total number of reqs in the RUNNING state should not
            # exceed the maximum number of reqs.
            num_new_seqs = task.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(task,
                                                      RequestStatus.BLOCKED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_reqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            blocked_pool.popleft()
            self._BlockIn(task, blocks_to_BlockIn)
            self._append_slots(task, blocks_to_copy)
            is_prefill = task.is_prefill()
            if is_prefill:
                prefill_tasks.append(
                    ScheduledTask(task,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_tasks.append(
                    ScheduledTask(task, token_chunk_size=1))
            budget.add_num_batched_tokens(task.task_id, num_new_tokens)
            budget.add_num_reqs(task.task_id, num_new_seqs)

        blocked_pool.extendleft(leftover_swapped)
        # add for scheduler profiling
        # swap_reqs = len(swapped_queue)
        # sched_reqs = swap_reqs - len(swapped_queue)
        # sched_to_swap_in = [s.task.task_id for s in decode_seq_groups]
        return blocked_pool, SchedulerBlockInOutputs(
            decode_tasks=decode_tasks,
            prefill_tasks=prefill_tasks,
            blocks_to_blockIn=blocks_to_BlockIn,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_tasks=infeasible_tasks,
        )

    def _schedule_ready_task(
        self,
        ready_pool: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """调度就绪池中(prefill 阶段)的任务

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled tasks.

        Args:
            ready_pool: contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, task can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining ReadyPool after scheduling and
            SchedulerBlockInOutputs.
        """
        init_queue = ready_pool
        ignored_tasks: List[Task] = []
        tasks: List[Task] = []
        # 不对等待队列进行排序，因为假设它已经是有序的
        # 复制该队列以确保输入队列不会被修改
        ready_pool = deque([s for s in ready_pool])

        _remaining_ready_tasks: Deque[Task] = deque()
        while self._can_schedule_form_readyPool(time.time()) and ready_pool:
            task = ready_pool[0]

            ready_reqs = task.get_reqs(status=RequestStatus.READY)
            assert len(ready_reqs) == 1, (
                "Ready task should have only one prompt request.")
            num_new_tokens = self._get_num_new_tokens(task,
                                                      RequestStatus.READY,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = ready_reqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            if num_new_tokens > self.prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens,
                    self.prompt_limit)
                for seq in ready_reqs:
                    seq.status = RequestStatus.FINISHED_IGNORED
                ignored_tasks.append(task)
                ready_pool.popleft()
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
                for seq in ready_reqs:
                    seq.status = RequestStatus.FINISHED_IGNORED
                ignored_tasks.append(task)
                ready_pool.popleft()
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
                    _remaining_ready_tasks.appendleft(task)
                    ready_pool.popleft()
                    continue

            num_new_seqs = task.get_task_running_reqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_reqs=num_new_seqs)):
                break

            # 可以调度这个序列
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            ready_pool.popleft()
            self._allocate_and_set_running(task) # 实际分配
            tasks.append(
                ScheduledTask(task=task,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(task.task_id, num_new_tokens)
            budget.add_num_reqs(task.task_id, num_new_seqs)

        # 将无法被调度的请求重新加入队列中
        ready_pool.extendleft(_remaining_ready_tasks)
        if len(tasks) > 0:
            self._last_schedule_formReadyPool = True

        # add for scheduler profiling
        # wait_reqs = len(init_queue)
        # sched_reqs = wait_reqs - len(waiting_queue)
        # sched_to_run = [s.task.task_id for s in prefills.seq_groups]
        # sched_to_ingore = [s.task.task_id for s in prefills.ignored_seq_groups]

        return ready_pool, SchedulerPrefillOutputs(
            tasks=tasks,
            ignored_tasks=ignored_tasks,
            _lookahead_slots_num=self._get_num_lookahead_slots(is_prefill=True))

    def _schedule_without_cp(self) -> SchedulerOutputs:
        """
        running队列和blocked队列全是decode阶段(no chunked prefill)
        任务池优先级: ReadyPool > RunningPool > BlockedPool
        此策略旨在优化吞吐量。首先，它将尽可能多的预填充请求进行批处理。
        然后它调度解码请求。如果MLU内存出现压力，解码请求可以被交换或抢占
        """ 
        # 调度预算：主要关注tokens总数和seq总数
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens, # 最大：2048
            max_num_reqs=self.scheduler_config.max_num_reqs,
        )
        # 确保在调度prefill请求之前包含当前运行中的序列数，  
        # 以避免预填充的调度超出 `max_num_seqs` 的限制。
        for task in self.RunningPool:
            budget.add_num_reqs(task.task_id,
                                task.get_task_running_reqs())
        curr_loras = set(
            task.lora_int_id
            for task in self.RunningPool) if self.lora_enabled else None # 启用了LoRA

        remaining_ready, scheduler_prefill = (self.ReadyPool,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, scheduler_running = (
            self.RunningPool, SchedulerRunningOutputs.create_empty())
        remaining_blocked, scheduler_blockIn = (
            self.BlockedPool, SchedulerBlockInOutputs.create_empty())

        # 如果阻塞池为空，调度就绪池任务
        if not self.BlockedPool:
            remaining_ready, scheduler_prefill = self._schedule_ready_task(
                self.ReadyPool, budget, curr_loras, enable_chunking=False)

        # scheduler_policy = Profilers.get_profiler(_name="fcfs")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v1")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v2")
        scheduler_policy = Profilers.get_profiler(_name="tas_v3")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v4")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v5")
        # 如果已经调度了预填充请求，则不要再调度解码请求
        # 如果 _schedule_prefills 未启用分块模式（chunking），
        # 那么self.running 中仅包含解码请求，而不包含分块的预填充请求
        if len(scheduler_prefill.tasks) == 0: # 没有调度就绪池的任务
            remaining_running, scheduler_running = self._schedule_running_task(
                self.RunningPool,
                budget,
                curr_loras,
                scheduler_policy,
                enable_chunking=False)

            # 如果有任何序列组被抢占或换出，则不要swap in任何序列组，  
            # 因为这意味着没有可用的插槽用于新的运行请求（running_scheduled.preempted存的是重计算的SG,running_scheduled.swapped_out存的是换出的SG）
            if len(scheduler_running.preempted) + len(
                    scheduler_running.swapped_out) == 0:
                remaining_blocked, scheduler_blockIn = self._schedule_blocked_task(
                    self.BlockedPool, budget, curr_loras, scheduler_policy)
        # 确保预算内的token数和序列数没有超出限制
        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_reqs <= self.scheduler_config.max_num_reqs

        # Update
        self.ReadyPool = remaining_ready
        self.ReadyPool.extendleft(scheduler_running.preempted) # 重计算的task加入到ReadyPool最左侧
        # 
        self.RunningPool = remaining_running
        self.RunningPool.extend([s.task for s in scheduler_prefill.tasks])
        self.RunningPool.extend(
            [s.task for s in scheduler_running.decode_tasks])
        self.RunningPool.extend(
            [s.task for s in scheduler_blockIn.decode_tasks])
        # 
        self.BlockedPool = remaining_blocked
        self.BlockedPool.extend(scheduler_running.swapped_out) # 换出的SG加入到swapped队列

        # 没开启chunked prefill时，running队列和blocked队列都不应该包含prefill阶段的task
        assert len(scheduler_running.prefill_tasks) == 0
        assert len(scheduler_blockIn.prefill_tasks) == 0
        # add for scheduler profiling
        self.schedule_step += 1
        # 包装一下调度结果并返回
        return SchedulerOutputs(
            scheduled_tasks=(scheduler_prefill.tasks +
                                  scheduler_running.decode_tasks +
                                  scheduler_blockIn.decode_tasks),
            num_prefill_tasks=len(scheduler_prefill.tasks),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=scheduler_blockIn.blocks_to_blockIn,
            blocks_to_swap_out=scheduler_running.blocks_to_swap_out,
            blocks_to_copy=merge_dicts(scheduler_running.blocks_to_copy,
                                       scheduler_blockIn.blocks_to_copy),
            ignored_tasks=scheduler_prefill.ignored_tasks +
            scheduler_blockIn.infeasible_tasks,
            num_lookahead_slots=scheduler_running.num_lookahead_slots,
            running_queue_size=len(self.RunningPool),
        )

    def _schedule_with_cp(self):
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
            max_num_reqs=self.scheduler_config.max_num_reqs,
        )
        curr_loras: Set[int] = set()

        remaining_ready, prefills = (self.ReadyPool,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.RunningPool, SchedulerRunningOutputs.create_empty())
        remaining_blocked, swapped_in = (
            self.BlockedPool, SchedulerBlockInOutputs.create_empty())

        # scheduler_policy = Profilers.get_profiler(_name="fcfs")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v1")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v2")
        scheduler_policy = Profilers.get_profiler(_name="tas_v3")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v4")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v5")
        remaining_running, running_scheduled = self._schedule_running_task(
            self.RunningPool,
            budget,
            curr_loras,
            scheduler_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            remaining_blocked, swapped_in = self._schedule_blocked_task(
                self.BlockedPool, budget, curr_loras, scheduler_policy)

        # Schedule new prefills.
        remaining_ready, prefills = self._schedule_ready_task(
            self.ReadyPool, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_reqs <= self.scheduler_config.max_num_reqs

        # Update waiting requests.
        self.ReadyPool = remaining_ready
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
        self.BlockedPool = remaining_blocked
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
            blocks_to_swap_in=swapped_in.blocks_to_blockIn,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=merge_dicts(running_scheduled.blocks_to_copy,
                                       swapped_in.blocks_to_copy),
            ignored_tasks=prefills.ignored_tasks,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.RunningPool),
        )

    def _schedule_entrance(self) -> SchedulerOutputs:
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_with_cp()
        else:
            return self._schedule_without_cp()

    def _MLU_memory_sufficient(self, task: Task) -> bool:
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

        return self.block_manager.MLU_memory_sufficient(
            task=task,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )

    def schedule_entrance(self) -> Tuple[List[TaskMetadata], SchedulerOutputs]:
        scheduler_outputs = self._schedule_entrance()
        now = time.time()

        task_metadata_list: List[TaskMetadata] = []
        # 依次遍历每一个序列组
        for i, scheduled_task in enumerate(
                scheduler_outputs.scheduled_tasks):
            task = scheduled_task.task
            token_chunk_size = scheduled_task.token_chunk_size
            task.maybe_set_first_scheduled_time(now)

            # req_id -> SequenceData
            req_data: Dict[int, RequestData] = {}
            # req_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}
            # 依次遍历task中的每一个req
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
                reqs = task.get_reqs()
                assert len(reqs) == 1  # Prefill 阶段只有一个seq
                # chunked prefill不需要进行采样。  
                if (token_chunk_size + reqs[0].data.get_num_computed_tokens() <
                        reqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            is_prompt = task.is_prefill()
            task_metadata = TaskMetadata(
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
            task_metadata_list.append(task_metadata)

        # 假设批次中的所有块都会在下次调度调用之前完成计算
        # 这是因为模型执行中的失败会导致 SInfer 实例崩溃，而不会进行重试
        for scheduled_task in scheduler_outputs.scheduled_tasks:
            self.block_manager.mark_blocks_as_computed(
                scheduled_task.task)

        return task_metadata_list, scheduler_outputs

    def fork_req(self, parent_req: Request, child_req: Request) -> None:
        self.block_manager.fork(parent_req, child_req)

    def free_req(self, req: Request) -> None:
        self.block_manager.free(req)

    def free_all_finished_tasks(self) -> None:
        finished_tasks_ = [task for task in self.RunningPool
                                        if task.is_finished()]

        self.RunningPool = deque(task for task in self.RunningPool
                                if not task.is_finished())

        self.running_tasks = len(self.RunningPool)
        self.ready_tasks = len(self.ReadyPool)
        self.blocked_tasks = len(self.BlockedPool)
        self.finished_tasks += len(finished_tasks_)
        self.finished_tasks_list += finished_tasks_

        total_seqs_ = self.running_tasks + self.ready_tasks + self.blocked_tasks + self.finished_tasks
        if total_seqs_ > self.total_tasks:
            self.total_tasks = total_seqs_

        self.batch_utils = self.running_tasks / self.scheduler_config.max_num_reqs
        self.block_utils = (self.block_manager._total_mlu_blocks_num -
                            self.block_manager.get_num_free_mlu_blocks()) / self.block_manager._total_mlu_blocks_num
        self.preempt_ratio = self.from_running_to_ready_tasks / self.total_tasks
        # tas_information.csv
        df_ = pd.DataFrame(
            [[self.running_tasks,self.ready_tasks, 
            len(finished_tasks_), self.batch_utils, self.block_utils]],
            columns=['running','ready','finished','batch_utils', 'block_utils'],
            index=[str(self.schedule_step)])
        self.df = pd.concat([self.df, df_])

    def _allocate_and_set_running(self, task: Task) -> None:
        self.block_manager.allocate(task)
        for req in task.get_reqs(status=RequestStatus.READY):
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
        # 遍历这个task中状态为running的所有req
        for req in task.get_reqs(status=RequestStatus.RUNNING):
            cows = self.block_manager.append_slots(req, num_lookahead_slots)

            for src, dests in cows.items():
                if src not in blocks_to_copy:
                    blocks_to_copy[src] = []
                blocks_to_copy[src].extend(dests)

    def _preempt(
        self,
        task: Task,
        blocks_to_blockout: Dict[int, int],
        preemption_strategy: Optional[PreemptionStrategy] = None,
    ) -> PreemptionStrategy:
        # 如果未指定抢占策略，将根据以下规则确定策略：
        # 默认使用重计算，因为其开销比交换模式更低。然而，当序列组包含多个序列（例如，束搜索）时，使用交换模式。
        # 缺陷：被交换的序列优先于等待的序列，所以包含多个序列的序列组隐式地优先于仅包含单个序列的序列组。
        # 为包含多个序列的序列组支持重计算。这需要更复杂的CUDA内核支持
        if preemption_strategy is None:
            if task.get_task_running_reqs() == 1:
                preemption_strategy = PreemptionStrategy.RECOMPUTE
            else:
                preemption_strategy = PreemptionStrategy.BLOCK
        if preemption_strategy == PreemptionStrategy.RECOMPUTE:
            self._preempt_by_recompute(task)
        elif preemption_strategy == PreemptionStrategy.BLOCK:
            self._preempt_by_block(task, blocks_to_blockout)
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
            req.status = RequestStatus.READY 
            # 释放这条seq对应的物理块
            # 即将对应物理块的引用-1，如果此时引用数量为0，说明对应物理块完全自由了，需要再将其放入自由物理块列表中
            self.free_req(req)
            req.reset_state_for_recompute()

    def _preempt_by_block(
        self,
        task: Task,
        blocks_to_blockout: Dict[int, int],
    ) -> None:
        # 释放该seq_group下所有seq的物理块，并为其分配对应的cpu物理块，
        # 将seq的状态从running改成swapped
        self._swap_out(task, blocks_to_blockout)

    def _BlockIn(
        self,
        task: Task,
        blocks_to_blockin: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.Block_in(task)
        blocks_to_blockin.update(mapping)
        for req in task.get_reqs(status=RequestStatus.BLOCKED):
            req.status = RequestStatus.RUNNING

    def _swap_out(
        self,
        task: Task,   # 需要被swap到cpu上的task
        blocks_to_blockout: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(task):
            # 检查是否可以将当前task对应的物理块swap到cpu上
            # 可以的条件：当前task占用的物理块数量 <= cpu上可用的物理块数量
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        # 释放该task下所有req的mlu物理块，并为其创建对应的cpu块
        # mapping：{mlu物理块id：cpu物理块id}
        mapping = self.block_manager.swap_out(task)
        blocks_to_blockout.update(mapping)
        for req in task.get_reqs(status=RequestStatus.RUNNING):
            req.status = RequestStatus.BLOCKED # 修改该task下所有req的状态：从running改成swapped

    def _can_schedule_form_readyPool(self, now: float) -> bool:
        """
        判断当下是否可以从就绪池中调度
        确保在调度过程中不会频繁地处理新来的task
        
        Args:
            now: 当前调度时间点
        """
        if self._last_schedule_formReadyPool:
            self._latency_lastFromReadyPool = now - self._last_schedule_time
        # 用当前调度时间更新prev_time
        # 由于目前还不知道本次是否会从就绪池中调度，因此_last_schedule_formReadyPool先设为False
        self._last_schedule_time, self._last_schedule_formReadyPool = now, False
        # Delay scheduling prompts to let readyPool fill up
        # delay_factor：用户配置的，用于调整调度间隔阈值的因子。大于0则意味着用户想开启阈值判断
        if self.scheduler_config.latency_factor > 0 and self.ReadyPool:
            # 计算在就绪池中最早到达的task的到达时间
            earliest_arrival_time = min(
                [e.metadata.arrival_time for e in self.ReadyPool])
            # 实际等待的时间 > 应该等待时间
            _can_sche = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.latency_factor * self._latency_lastFromReadyPool)
                or not self.RunningPool)
        else:
            _can_sche = True
        return _can_sche

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
                'TTFT/s': [],  # 首次 token 时间
                'TPOT/s': [],  # 每个 token 的解码时间
                'e2e_time/s': [],  # 任务的端到端时间
                'time_in_pool/s': [],  # 任务在池中的等待时间
                'context_latency/s': [],  # 上下文延迟
                'total_decoding_time/s': [],  # 总解码时间
                'generate_tokens': []  # 生成的 token 数量
            },
            dtype=np.float32
        )
        for task in tasks:
            ttft = task.metadata.first_token_time - task.metadata.arrival_time
            time_in_pool = task.metadata.time_in_pool
            context_latency = task.metadata.first_token_time - task.metadata.first_scheduled_time # 任务首次调度到首次生成 token 的时间
            total_decoding_time = task.metadata.finished_time - task.metadata.first_token_time
            generate_token_num = task.get_reqs()[0].get_output_len() - 1
            tpot = total_decoding_time if generate_token_num == 0 \
                                                else total_decoding_time / generate_token_num
            e2e_time = total_decoding_time + ttft
            df_ = pd.DataFrame(
                [[ttft, tpot, time_in_pool, e2e_time,context_latency, total_decoding_time, generate_token_num]],
                columns=['TTFT/s', 'TPOT/s', 'e2e_time/s','time_in_pool/s', 'context_latency/s', 'total_decoding_time/s', 'generate_tokens'],
                index=[str(task.task_id)]
            )
            df = pd.concat([df, df_])
        max_, min_, mean_, p99_ = df.max(), df.min(), df.mean(), df.quantile(0.90)
        df.loc['Max'] = max_
        df.loc['Min'] = min_
        df.loc['Mean'] = mean_
        df.loc['P99'] = p99_
        return df

    def __del__(self):
        logger.info(f"TAScheduler profiling saving...")
        figure = self.df.plot(y=['ready', 'running', 'finished']).get_figure()
        figure.savefig("TAS_TaskPool.jpg", dpi=300)
        plt.close(figure)
        figure = self.df.plot(y=['batch_utils', 'block_utils']).get_figure()
        figure.savefig("TAS_ResourceUtil_Rate.jpg", dpi=300)
        plt.close(figure)

        time_df = self.summary_finished_tasks(self.finished_tasks_list)

        max_, min_ , mean_ = self.df.max(), self.df.min(), self.df.mean()
        self.df.loc["Max"] = max_
        self.df.loc["Min"] = min_
        self.df.loc["Mean"] = mean_
        with pd.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.max_colwidth', None,
                            'display.float_format', '{:^6,.2f}'.format,
                            'expand_frame_repr', False):
            # logger.info(self.df.loc[["Max", "Min", "Mean"]])
            logger.info(time_df.loc[["Max", "Min", "Mean", "P99"]])
        self.df.astype(str).to_csv("tas_infomation.csv", mode="w")
        time_df.astype(str).to_csv("sinfer_perf.csv", mode="w")
