from collections import deque
from typing import Deque

from sinfer.task import Task


class Policy:
    def sort_by_priority(
        self,
        now: float,
        tasks: Deque[Task],
    ) -> Deque[Task]:
        return deque(
            sorted(
                tasks,
                key=lambda seq_group: self.get_score(now, seq_group),
                reverse=True,
            ))
    
    def get_score(
        self,
        now: float,
        task: Task,
    ) -> float:
        raise NotImplementedError


class FCFS(Policy):
    def get_score(
        self,
        now: float,
        task: Task,
    ) -> float:
        return now - task.metrics.arrival_time

class TAS(Policy):
    '''
    Task-Aware Scheduler
    priority = α * (now - arrival_time) + β * time_in_queue - γ * (first_scheduled_time) + δ * (execution_progress)
    '''
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 5.0, delta: float = 1.0):
        """
        初始化 TAS 策略的权重参数
        alpha: 控制到达时间对优先级的影响
        beta: 控制队列等待时间对优先级的影响
        gamma: 控制已调度任务的优先级
        delta: 控制执行进度对优先级的影响(生成进度)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def get_score(
        self,
        now: float,
        task: Task,
    ) -> float:
        """
        根据任务的各种属性计算优先级
        :param now: 当前时间
        :param task: 任务实例
        :return: 优先级值
        """
        metrics = task.metrics

        arrival_priority = now - metrics.arrival_time # 任务等待的总时间

        # 队列等待时间的影响
        time_in_queue = metrics.time_in_queue or 0.0

        # 首次调度时间的影响（降低已经被调度任务的优先级）
        first_scheduled_penalty = 1.0 if metrics.first_scheduled_time else 0.0

        # 执行进度的影响（以生成令牌的时间为参考）
        execution_progress = (metrics.last_token_time or 0.0) - (metrics.first_token_time or 0.0)

        # 计算优先级（加权公式）
        score = (
            self.alpha * arrival_priority +
            self.beta * time_in_queue -
            self.gamma * first_scheduled_penalty +
            self.delta * execution_progress
        )
        return score

class PolicyFactory:

    _POLICY_REGISTRY = {
        'tas' : TAS,
        'fcfs': FCFS,
        }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
