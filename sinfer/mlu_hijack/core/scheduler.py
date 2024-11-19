import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
from sinfer.config import CacheConfig, LoRAConfig, SchedulerConfig
from sinfer.core.policy import Policy
from sinfer.core.scheduler import SchedulingBudget, SchedulerPrefillOutputs, SchedulerRunningOutputs, SchedulerSwappedInOutputs, SchedulerOutputs, Scheduler
from sinfer.sequence import SequenceGroup, Sequence
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.logger import init_logger

logger = init_logger(__name__)


sinfer__core__scheduler__Scheduler____init____org = Scheduler.__init__
sinfer__core__scheduler__Scheduler___schedule_prefills__org = Scheduler._schedule_prefills
sinfer__core__scheduler__Scheduler___schedule_running__org = Scheduler._schedule_running
sinfer__core__scheduler__Scheduler___schedule_swapped__org = Scheduler._schedule_swapped
sinfer__core__scheduler__Scheduler___schedule_default__org = Scheduler._schedule_default


def sinfer__core__scheduler__Scheduler____init__(
    self,
    scheduler_config: SchedulerConfig,
    cache_config: CacheConfig,
    lora_config: Optional[LoRAConfig],
) -> None:
    sinfer__core__scheduler__Scheduler____init____org(
        self=self,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        lora_config=lora_config
    )
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    logger.info(f"SInfer scheduler profiling start...")
    
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

    self.finished_seq_groups = []
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def sinfer__core__scheduler__Scheduler___schedule_prefills(
    self,
    waiting_queue: deque,
    budget: SchedulingBudget,
    curr_loras: Optional[Set[int]],
    enable_chunking: bool = False,
) -> Tuple[deque, SchedulerPrefillOutputs]:
    remaining_waiting, prefills = sinfer__core__scheduler__Scheduler___schedule_prefills__org(
        self=self,
        waiting_queue=waiting_queue,
        budget=budget,
        curr_loras=curr_loras,
        enable_chunking=enable_chunking
    )
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    # wait_reqs = len(waiting_queue)
    # sched_reqs = wait_reqs - len(remaining_waiting)
    # sched_to_run = [s.seq_group.request_id for s in prefills.seq_groups]
    # sched_to_ingore = [s.seq_group.request_id for s in prefills.ignored_seq_groups]
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return remaining_waiting, prefills


def sinfer__core__scheduler__Scheduler___schedule_running(
    self,
    running_queue: deque,
    budget: SchedulingBudget,
    curr_loras: Optional[Set[int]],
    policy: Policy,
    enable_chunking: bool = False,
) -> Tuple[deque, SchedulerRunningOutputs]:
    remaining_running, running_scheduled = sinfer__core__scheduler__Scheduler___schedule_running__org(
        self=self,
        running_queue=running_queue,
        budget=budget,
        curr_loras=curr_loras,
        policy=policy,
        enable_chunking=enable_chunking
    )
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    # run_reqs = len(running_queue)
    # sched_reqs = run_reqs - len(remaining_running)
    sched_to_preempt = [s.request_id for s in running_scheduled.preempted]
    # sched_to_swap_out = [s.request_id for s in running_scheduled.swapped_out]
    self.running_to_waiting_seqs += len(sched_to_preempt)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return remaining_running, running_scheduled


def sinfer__core__scheduler__Scheduler___schedule_swapped(
    self,
    swapped_queue: deque,
    budget: SchedulingBudget,
    curr_loras: Optional[Set[int]],
    policy: Policy,
    enable_chunking: bool = False,
) -> Tuple[deque, SchedulerSwappedInOutputs]:
    remaining_swapped, swapped_in = sinfer__core__scheduler__Scheduler___schedule_swapped__org(
        self=self,
        swapped_queue=swapped_queue,
        budget=budget,
        curr_loras=curr_loras,
        policy=policy,
        enable_chunking=enable_chunking
    )
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    # swap_reqs = len(swapped_queue)
    # sched_reqs = swap_reqs - len(remaining_swapped)
    # sched_to_swap_in = [s.seq_group.request_id for s in swapped_in.decode_seq_groups]
    # sched_to_ignore = [s.seq_group.request_id for s in swapped_in.infeasible_seq_groups]
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return remaining_swapped, swapped_in


def sinfer__core__scheduler__Scheduler___schedule_default(self) -> SchedulerOutputs:
    scheduler_outputs = sinfer__core__scheduler__Scheduler___schedule_default__org(self)
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    self.sched_step += 1
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return scheduler_outputs


def sinfer__core__scheduler__Scheduler__free_finished_seq_groups(self) -> None:
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    finished_seq_groups_ = [seq_group for seq_group in self.running
                                    if seq_group.is_finished()]

    self.running = deque(seq_group for seq_group in self.running
                            if not seq_group.is_finished())

    self.running_seqs = len(self.running)
    self.waiting_seqs = len(self.waiting)
    self.swapped_seqs = len(self.swapped)
    self.finished_seqs += len(finished_seq_groups_)
    self.finished_seq_groups += finished_seq_groups_

    total_seqs_ = self.running_seqs + self.waiting_seqs + self.swapped_seqs + self.finished_seqs
    if total_seqs_ > self.total_seqs:
        self.total_seqs = total_seqs_

    self.batch_utils = self.running_seqs / self.scheduler_config.max_num_seqs
    self.block_utils = (self.block_manager.num_total_gpu_blocks -
                        self.block_manager.get_num_free_gpu_blocks()) / self.block_manager.num_total_gpu_blocks
    self.preempt_ratio = self.running_to_waiting_seqs / self.total_seqs

    df_ = pd.DataFrame(
        [[self.waiting_seqs, self.running_seqs, self.swapped_seqs,
          len(finished_seq_groups_), self.batch_utils, self.block_utils, self.preempt_ratio]],
        columns=['waiting', 'running', 'swapped',
                 'finished','batch_utils', 'block_utils', 'preempt_ratio'],
        index=[str(self.sched_step)])
    self.df = pd.concat([self.df, df_])
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def summary_finished_seq_groups(seq_groups: List[SequenceGroup]):
    df = pd.DataFrame(
        data={
            'ttft/s': [], 'time_in_queue/s': [], 'context_latency/s': [], 'decoder_latency/s': []
        },
        dtype=np.float32
    )
    for seq_group in seq_groups:
        ttft = seq_group.metrics.first_token_time - seq_group.metrics.arrival_time
        time_in_queue = seq_group.metrics.time_in_queue
        context_latency = seq_group.metrics.first_token_time - seq_group.metrics.first_scheduled_time
        decoder_latency = seq_group.metrics.finished_time - seq_group.metrics.first_token_time
        decoder_token_num = seq_group.get_seqs()[0].get_output_len() - 1
        per_token_latency = decoder_latency if decoder_token_num == 0 \
                                            else decoder_latency / decoder_token_num
        df_ = pd.DataFrame(
            [[ttft, time_in_queue, context_latency, decoder_latency, per_token_latency, decoder_token_num]],
            columns=['ttft/s', 'time_in_queue/s', 'context_latency/s', 'decoder_latency/s', 'per_token_latency/s', 'decoder_tokens'],
            index=[str(seq_group.request_id)]
        )
        df = pd.concat([df, df_])
    sum_, max_, mean_, min_, p99_ = df.sum(), df.max(), df.mean(), df.min(), df.quantile(0.99)
    df.loc['Sum'] = sum_
    df.loc['Max'] = max_
    df.loc['Mean'] = mean_
    df.loc['Min'] = min_
    df.loc['P99'] = p99_
    return df


def sinfer__core__scheduler__Scheduler____del__(self):
    logger.info(f"SInfer scheduler profiling save...")
    figure = self.df.plot(y=['waiting', 'running', 'swapped', 'finished']).get_figure()
    figure.savefig("sinfer_queue_view.jpg", dpi=300)
    plt.close(figure)
    figure = self.df.plot(y=['batch_utils', 'block_utils', 'preempt_ratio']).get_figure()
    figure.savefig("sinfer_utilization_view.jpg", dpi=300)
    plt.close(figure)

    time_df = summary_finished_seq_groups(self.finished_seq_groups)

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


MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler.__init__,
                                  sinfer__core__scheduler__Scheduler____init__)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler._schedule_prefills,
                                  sinfer__core__scheduler__Scheduler___schedule_prefills)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler._schedule_running,
                                  sinfer__core__scheduler__Scheduler___schedule_running)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler._schedule_swapped,
                                  sinfer__core__scheduler__Scheduler___schedule_swapped)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler._schedule_default,
                                  sinfer__core__scheduler__Scheduler___schedule_default)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler.free_finished_seq_groups,
                                  sinfer__core__scheduler__Scheduler__free_finished_seq_groups)
MluHijackObject.add_hijack_object(Scheduler,
                                  "__del__",
                                  sinfer__core__scheduler__Scheduler____del__)