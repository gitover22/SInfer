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
from sinfer.scheduler.TASfactory import Profiler, Profilers
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
    task: Task
    token_chunk_size: int

@dataclass
class SchedulerOutputs:
    scheduled_tasks: Iterable[ScheduledTask]
    num_prefill_tasks: int 
    num_batched_tokens: int 
    blocks_to_swap_in: Dict[int, int]
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    ignored_tasks: List[Task]
    num_lookahead_slots: int
    running_queue_size: int

    def __post_init__(self):
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
    decode_tasks: List[Task]    
    prefill_tasks: List[Task]
    preempted: List[Task]       
    swapped_out: List[Task]    
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
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
    decode_tasks: List[Task]
    prefill_tasks: List[Task]
    blocks_to_blockIn: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    num_lookahead_slots: int
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
    tasks: List[Task]
    ignored_tasks: List[Task]
    _lookahead_slots_num: int
    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            tasks=[],
            ignored_tasks=[],
            _lookahead_slots_num=0,
        )

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
            
        self._latency_lastFromReadyPool = 0.0
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

        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        
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

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_req(self) -> int:
        return 1

    def add_task(self, task: Task) -> None:
        self.ReadyPool.append(task)

    def abort_task(self, task_id: Union[str, Iterable[str]]) -> None:
        if isinstance(task_id, str):
            task_id = (task_id, )
        task_ids = set(task_id)
        for state_queue in [self.ReadyPool, self.RunningPool, self.BlockedPool]:
            aborted_tasks: List[Task] = []
            for task in state_queue:
                if not task_ids:
                    break
                if task.task_id in task_ids:
                    aborted_tasks.append(task)
                    task_ids.remove(task.task_id)
            for aborted_task in aborted_tasks:
                state_queue.remove(aborted_task)
                for seq in aborted_task.get_reqs():
                    if seq.is_finished():
                        continue
                    seq.status = RequestStatus.FINISHED_ABORTED
                    self.free_req(seq)

    def has_unfinished_tasks(self) -> bool:
        return len(self.ReadyPool) != 0 or len(self.RunningPool) != 0 or len(
            self.BlockedPool) != 0

    def get_num_unfinished_tasks(self) -> int:
        return len(self.ReadyPool) + len(self.RunningPool) + len(self.BlockedPool)

    def _schedule_running_task(
        self,
        running_pool: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Profiler,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        init_queue = running_pool
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
                    victim_seq_group = running_pool.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_BlockOut)
                    if preempted_mode == PreemptionStrategy.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                else:
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
                if enable_chunking:
                    num_running_seqs = task.get_max_num_running_seqs()
                    budget.add_num_reqs(task.task_id, num_running_seqs)
                if curr_loras is not None and task.lora_int_id > 0:
                    curr_loras.add(task.lora_int_id)
        sched_to_preempt = [s.task_id for s in preempted]
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
                    leftover_swapped.appendleft(task)
                    blocked_pool.popleft()
                    continue

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
                    ScheduledTask(task,token_chunk_size=num_new_tokens))
            else:
                decode_tasks.append(
                    ScheduledTask(task, token_chunk_size=1))
            budget.add_num_batched_tokens(task.task_id, num_new_tokens)
            budget.add_num_reqs(task.task_id, num_new_seqs)

        blocked_pool.extendleft(leftover_swapped)
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
        init_queue = ready_pool
        ignored_tasks: List[Task] = []
        tasks: List[Task] = []
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
                    _remaining_ready_tasks.appendleft(task)
                    ready_pool.popleft()
                    continue

            num_new_seqs = task.get_task_running_reqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_reqs=num_new_seqs)):
                break

            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            ready_pool.popleft()
            self._allocate_and_set_running(task)
            tasks.append(
                ScheduledTask(task=task,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(task.task_id, num_new_tokens)
            budget.add_num_reqs(task.task_id, num_new_seqs)

        ready_pool.extendleft(_remaining_ready_tasks)
        if len(tasks) > 0:
            self._last_schedule_formReadyPool = True

        return ready_pool, SchedulerPrefillOutputs(
            tasks=tasks,
            ignored_tasks=ignored_tasks,
            _lookahead_slots_num=self._get_num_lookahead_slots(is_prefill=True))

    def _schedule_without_cp(self) -> SchedulerOutputs:
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_reqs=self.scheduler_config.max_num_reqs,
        )
        for task in self.RunningPool:
            budget.add_num_reqs(task.task_id,
                                task.get_task_running_reqs())
        curr_loras = set(
            task.lora_int_id
            for task in self.RunningPool) if self.lora_enabled else None 

        remaining_ready, scheduler_prefill = (self.ReadyPool,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, scheduler_running = (
            self.RunningPool, SchedulerRunningOutputs.create_empty())
        remaining_blocked, scheduler_blockIn = (
            self.BlockedPool, SchedulerBlockInOutputs.create_empty())

        if not self.BlockedPool:
            remaining_ready, scheduler_prefill = self._schedule_ready_task(
                self.ReadyPool, budget, curr_loras, enable_chunking=False)

        scheduler_policy = Profilers.get_profiler(_name="fcfs")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v1")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v2")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v3")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v4")
        # scheduler_policy = Profilers.get_profiler(_name="tas_v5")
        if len(scheduler_prefill.tasks) == 0: 
            remaining_running, scheduler_running = self._schedule_running_task(
                self.RunningPool,
                budget,
                curr_loras,
                scheduler_policy,
                enable_chunking=False)

            if len(scheduler_running.preempted) + len(
                    scheduler_running.swapped_out) == 0:
                remaining_blocked, scheduler_blockIn = self._schedule_blocked_task(
                    self.BlockedPool, budget, curr_loras, scheduler_policy)
        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_reqs <= self.scheduler_config.max_num_reqs

        self.ReadyPool = remaining_ready
        self.ReadyPool.extendleft(scheduler_running.preempted)
        self.RunningPool = remaining_running
        self.RunningPool.extend([s.task for s in scheduler_prefill.tasks])
        self.RunningPool.extend(
            [s.task for s in scheduler_running.decode_tasks])
        self.RunningPool.extend(
            [s.task for s in scheduler_blockIn.decode_tasks])
        # 
        self.BlockedPool = remaining_blocked
        self.BlockedPool.extend(scheduler_running.swapped_out)

        assert len(scheduler_running.prefill_tasks) == 0
        assert len(scheduler_blockIn.prefill_tasks) == 0
        self.schedule_step += 1
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

        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            remaining_blocked, swapped_in = self._schedule_blocked_task(
                self.BlockedPool, budget, curr_loras, scheduler_policy)

        remaining_ready, prefills = self._schedule_ready_task(
            self.ReadyPool, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_reqs <= self.scheduler_config.max_num_reqs
        
        self.ReadyPool = remaining_ready
        self.ReadyPool.extendleft(running_scheduled.preempted)
        
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
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        is_prefill = False

        return self.block_manager.MLU_memory_sufficient(
            task=task,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )

    def schedule_entrance(self) -> Tuple[List[TaskMetadata], SchedulerOutputs]:
        scheduler_outputs = self._schedule_entrance()
        now = time.time()

        task_metadata_list: List[TaskMetadata] = []

        for i, scheduled_task in enumerate(
                scheduler_outputs.scheduled_tasks):
            task = scheduled_task.task
            token_chunk_size = scheduled_task.token_chunk_size
            task.maybe_set_first_scheduled_time(now)

            req_data: Dict[int, RequestData] = {}
       
            block_tables: Dict[int, List[int]] = {}
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
                assert len(reqs) == 1 
                if (token_chunk_size + reqs[0].data.get_num_computed_tokens() <
                        reqs[0].data.get_len()):
                    do_sample = False

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
                multi_modal_data=task.multi_modal_data
                if scheduler_outputs.num_prefill_tasks > 0 else None,
            )
            task_metadata_list.append(task_metadata)

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

        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False)

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
        reqs = task.get_reqs(status=RequestStatus.RUNNING)
        assert len(reqs) == 1
        for req in reqs:
            req.status = RequestStatus.READY 
            self.free_req(req)
            req.reset_state_for_recompute()

    def _preempt_by_block(
        self,
        task: Task,
        blocks_to_blockout: Dict[int, int],
    ) -> None:
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
        task: Task,
        blocks_to_blockout: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(task):
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(task)
        blocks_to_blockout.update(mapping)
        for req in task.get_reqs(status=RequestStatus.RUNNING):
            req.status = RequestStatus.BLOCKED

    def _can_schedule_form_readyPool(self, now: float) -> bool:
        if self._last_schedule_formReadyPool:
            self._latency_lastFromReadyPool = now - self._last_schedule_time

        self._last_schedule_time, self._last_schedule_formReadyPool = now, False
        if self.scheduler_config.latency_factor > 0 and self.ReadyPool:
            earliest_arrival_time = min(
                [e.metadata.arrival_time for e in self.ReadyPool])
            _can_sche = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.latency_factor * self._latency_lastFromReadyPool)
                or not self.RunningPool)
        else:
            _can_sche = True
        return _can_sche

    def _get_num_lookahead_slots(self, is_prefill: bool) -> int:
        if is_prefill:
            return 0
        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_tokens(self, task: Task,
                            status: RequestStatus, enable_chunking: bool,
                            budget: SchedulingBudget) -> int:
        num_new_tokens = 0
        reqs = task.get_reqs(status=status)
        for req in reqs:
            num_new_tokens += req.get_num_new_tokens()
        assert num_new_tokens > 0
        if enable_chunking and len(reqs) == 1:
            num_new_tokens = min(num_new_tokens,
                                 budget.remaining_token_budget())
        return num_new_tokens
    
    def summary_finished_tasks(self,tasks: List[Task]):
        df = pd.DataFrame(
            data={
                'TTFT/s': [],
                'TPOT/s': [],
                'e2e_time/s': [], 
                'total_decoding_time/s': [], 
                'generate_tokens': [] 
            },
            dtype=np.float32
        )
        for task in tasks:
            ttft = task.metadata.first_token_time - task.metadata.arrival_time
            print(task.metadata.first_token_time)
            print(task.metadata.arrival_time)

            total_decoding_time = task.metadata.finished_time - task.metadata.first_token_time
            generate_token_num = task.get_reqs()[0].get_output_len() - 1
            tpot = total_decoding_time if generate_token_num == 0 \
                                                else total_decoding_time / generate_token_num
            e2e_time = total_decoding_time + ttft
            df_ = pd.DataFrame(
                [[ttft, tpot,e2e_time, total_decoding_time, generate_token_num]],
                columns=['TTFT/s', 'TPOT/s', 'e2e_time/s','total_decoding_time/s', 'generate_tokens'],
                index=[str(task.task_id)]
            )
            df = pd.concat([df, df_])
        max_, min_, mean_, p90_ = df.max(), df.min(), df.mean(), df.quantile(0.90)
        df.loc['Max'] = max_
        df.loc['Min'] = min_
        df.loc['Mean'] = mean_
        df.loc['P90'] = p90_
        return df

    def __del__(self):
        figure = self.df.plot(y=['ready', 'running', 'finished']).get_figure()
        plt.close(figure)
        figure = self.df.plot(y=['batch_utils', 'block_utils']).get_figure()
        plt.close(figure)

        time_df = self.summary_finished_tasks(self.finished_tasks_list)

        max_, min_ , mean_ = self.df.max(), self.df.min(), self.df.mean()
        self.df.loc["Max"] = max_
        self.df.loc["Min"] = min_
        self.df.loc["Mean"] = mean_
        if os.getenv("TASRESULT", "False").lower() == "true":
            with pd.option_context('display.max_rows', None,
                                'display.max_columns', None,
                                'display.max_colwidth', None,
                                'display.float_format', '{:^6,.3f}'.format,
                                'expand_frame_repr', False):
                logger.info(time_df.loc[["Max", "Min", "Mean", "P90"]])
