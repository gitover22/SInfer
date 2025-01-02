from collections import deque
from typing import Deque
import math

from sinfer.task import Task


class Profiler:
    def sort_by_score(
        self,
        current_time: float,
        tasks: Deque[Task],
    ) -> Deque[Task]:
        return deque(
            sorted(
                tasks,
                key=lambda task: self.total_score(current_time, task),
                reverse=True,
            ))
    def total_score(
        self,
        now: float,
        task: Task,
    ) -> float:
        raise NotImplementedError

class Profiler_FCFS(Profiler):
    def total_score(
        self,
        current_time: float,
        task: Task,
    ) -> float:
        metadata = task.metadata
        return current_time - metadata.arrival_time

class Profiler_TAS_v1(Profiler):
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 5.0, delta: float = 1.0, epsilon: float = 2.0):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
    def promptLen_score(self, promptLen: int) -> float:
        return promptLen / 100
    def total_score(
        self,
        current_time: float,
        task: Task,
    ) -> float:
        metadata = task.metadata

        arrival_priority = current_time - metadata.arrival_time

        time_in_queue = metadata.time_in_queue or 0.0

        first_scheduled_ = 1.0 if metadata.first_scheduled_time else 0.0

        execution_progress = (metadata.last_token_time or 0.0) - (metadata.first_token_time or 0.0)

        length_score = self.promptLen_score(metadata.promptLen)

        score = (
            self.alpha * arrival_priority +
            self.beta * time_in_queue +
            self.gamma * first_scheduled_ +
            self.delta * execution_progress -
            self.epsilon * length_score
        )
        return score

class Profiler_TAS_v2(Profiler):
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 5.0, delta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _arrival_priority(self, current_time: float, arrival_time: float) -> float:
        wait_time = current_time - arrival_time
        return self.alpha * (wait_time ** 0.5)

    def _queue_time_effect(self, time_in_queue: float) -> float:
        if time_in_queue < 10:
            return self.beta * (time_in_queue ** 2) 
        else:
            return self.beta * (10 + (time_in_queue - 10) * 1.5)

    def _schedule_penalty(self, first_scheduled_time: float) -> float:
        if first_scheduled_time:
            return self.gamma * (1 / (1 + first_scheduled_time)) 
        return 0.0

    def _execution_progress_effect(self, first_token_time: float, last_token_time: float) -> float:
        if first_token_time and last_token_time:
            progress = last_token_time - first_token_time
            return self.delta * (progress ** 0.7)
        return 0.0

    def _complex_weighting(self, arrival_priority: float, queue_effect: float, penalty: float, progress_effect: float) -> float:
        total_weight = abs(arrival_priority) + abs(queue_effect) + abs(penalty) + abs(progress_effect)
        if total_weight == 0:
            return 0.0
        normalized_arrival = arrival_priority / total_weight
        normalized_queue = queue_effect / total_weight
        normalized_penalty = penalty / total_weight
        normalized_progress = progress_effect / total_weight

        return (
            (self.alpha + normalized_arrival) * arrival_priority +
            (self.beta + normalized_queue) * queue_effect -
            (self.gamma + normalized_penalty) * penalty +
            (self.delta + normalized_progress) * progress_effect
        )

    def total_score(
        self,
        current_time: float,
        task: Task,
    ) -> float:
        metadata = task.metadata

        arrival_priority = self._arrival_priority(current_time, metadata.arrival_time)
        queue_effect = self._queue_time_effect(metadata.time_in_queue or 0.0)
        schedule_penalty = self._schedule_penalty(metadata.first_scheduled_time or 0.0)
        progress_effect = self._execution_progress_effect(metadata.first_token_time or 0.0, metadata.last_token_time or 0.0)

        score = self._complex_weighting(arrival_priority, queue_effect, schedule_penalty, progress_effect)

        return score

class Profiler_TAS_v3(Profiler):
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, 
                 gamma: float = 5.0, delta: float = 1.0, 
                 epsilon: float = 1.5,scale:int =100, polynomial_degree: int = 3):
        """
        初始化参数，并引入多项式阶数
        :param alpha: 到达时间得分权重
        :param beta: 队列等待时间得分权重
        :param gamma: 首次调度时间得分权重
        :param delta: 长请求惩罚系数
        :param epsilon: 优先级权重
        :param polynomial_degree: 多项式的阶数，用于调整复杂性
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.length_scale = 100
        self.polynomial_degree = polynomial_degree  # 抽象的多项式阶数

    def _polynomial_sum(self, x: float, degree: int) -> float:
        """
        通用的多项式求和方法
        :param x: 输入值
        :param degree: 多项式的阶数
        :return: 多项式求和结果
        """
        return sum([(x ** i) / math.factorial(i) for i in range(1, degree + 1)])

    def _arrival_score(self, current_time: float, arrival_time: float, priority: int) -> float:
        """
        根据到达时间和优先级计算得分
        """
        if arrival_time is None:
            return 0.0
        wait_time = current_time - arrival_time
        poly_sum = self._polynomial_sum(wait_time, self.polynomial_degree)  # 使用抽象多项式求和
        priority_factor = math.exp(0.2 * priority)  # 优先级的指数加权
        return self.alpha * poly_sum * priority_factor

    def _schedule_score(self, first_scheduled_time: float) -> float:
        """
        根据首次调度时间计算得分
        """
        if first_scheduled_time is None:
            return 0.0
        poly_sum = self._polynomial_sum(first_scheduled_time, self.polynomial_degree)  # 使用多项式求和
        bonus = math.sqrt(1 + first_scheduled_time)  # 首次调度时间的平方根奖励
        return self.gamma * poly_sum * bonus

    def promptLen_score(self, promptLen: int) -> float:
        """
        根据用户请求长度（Prompt 长度）计算得分
        """
        if promptLen is None or promptLen == 0:
            return 0.0
        poly_sum = self._polynomial_sum(promptLen/self.length_scale, self.polynomial_degree)  # 使用多项式求和
        penalty = math.log(1 + poly_sum)  # 对数惩罚
        return 1 / (1 + penalty)  # 长请求得分越低

    def time_in_queue_score(self, time_in_queue: float, priority: int) -> float:
        """
        根据任务在队列中的等待时间计算得分
        """
        if time_in_queue is None or time_in_queue == 0.0:
            return 0.0
        poly_sum = self._polynomial_sum(time_in_queue, self.polynomial_degree)  # 使用多项式求和
        priority_factor = math.exp(0.1 * priority)  # 优先级的指数加成
        return self.beta * poly_sum * priority_factor

    def total_score(
        self,
        current_time: float,
        task: Task,
    ) -> float:
        """
        计算任务的总得分
        """
        metadata = task.metadata
        priority = metadata.priority
        score = 0.0

        # Arrival Score
        if metadata.arrival_time != 0:
            arrival_time = metadata.arrival_time
            score += self._arrival_score(current_time, arrival_time, priority)
        else:
            metadata.arrival_time = current_time

        # Queue Score
        if metadata.time_in_queue is not None:
            time_in_queue = metadata.time_in_queue
            queue_score = self.time_in_queue_score(time_in_queue, priority)
            score += queue_score
        else:
            metadata.time_in_queue = 0.0

        # Schedule Score
        if metadata.first_scheduled_time is not None:
            schedule_score = self._schedule_score(metadata.first_scheduled_time)
            score += schedule_score

        # Prompt Length Score
        if metadata.prompt_length is not None:
            length_score = self.promptLen_score(metadata.prompt_length)
            score += length_score

        return score



class Profiler_TAS_v4(Profiler):
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, 
                 gamma: float = 5.0, delta: float = 1.0, epsilon: float = 1.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.length_score = 100
    def _arrival_score(self, current_time: float, arrival_time: float, priority: int) -> float:
        if arrival_time is None:
            return 0.0
        wait_time = current_time - arrival_time
        base_score = self.alpha * (1 / (1 + math.exp(-0.1 * wait_time))) 
        return base_score * (self.epsilon if priority > 0 else 1.0) 
    def _schedule_score(self, first_scheduled_time: float) -> float:
        if first_scheduled_time:
            return self.gamma * (1 / (1 + first_scheduled_time)) 
        return 0.0
    def promptLen_score(self, promptLen: int) -> float:
        return promptLen / self.length_score
    def time_in_queue_score(self, time_in_queue: float, priority: int) -> float:
        if time_in_queue is None or time_in_queue == 0.0:
            return 0.0
        priority_factor = self.epsilon if priority > 0 else 1.0
        return self.beta * (2 * (time_in_queue ** 0.5)) * priority_factor
    def total_score(
        self,
        current_time: float,
        task: Task,
    ) -> float:
        metadata = task.metadata
        priority = metadata.priority
        score = 0.0
        if metadata.arrival_time != 0:
            arrival_time = metadata.arrival_time
            score += self._arrival_score(current_time, arrival_time, priority)
        else:
            metadata.arrival_time = current_time
        if metadata.time_in_queue is not None:
            time_in_queue = metadata.time_in_queue
            queue_score = self.time_in_queue_score(time_in_queue, priority)
            score += queue_score
        else:
            metadata.time_in_queue = 0.0
        if metadata.first_scheduled_time is not None:
            schedule_score = self._schedule_score(metadata.first_scheduled_time)
            score += schedule_score
        if metadata.prompt_length is not None:
            length_score = self.promptLen_score(metadata.prompt_length)
            score += length_score
        return score

class Profiler_TAS_v5(Profiler):
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, 
                 gamma: float = 5.0, delta: float = 1.0, epsilon: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

    def _arrival_priority(self, current_time: float, arrival_time: float, priority: int) -> float:
        if arrival_time is None:
            return 0.0
        wait_time = current_time - arrival_time
        base_score = self.alpha * (1 / (1 + math.exp(-0.1 * wait_time))) 
        return base_score * (1.5 if priority > 0 else 1.0) 

    def _queue_time_effect(self, time_in_queue: float, priority: int) -> float:
        if time_in_queue is None or time_in_queue == 0.0:
            return 0.0
        base_score = self.beta * (2 * (time_in_queue ** 0.5))
        return base_score * (1.2 if priority > 0 else 1.0)

    def _schedule_bonus(self, first_scheduled_time: float) -> float:
        if first_scheduled_time is None:
            return 0.0
        return self.gamma * (1 / (1 + first_scheduled_time)) 

    def _execution_progress_effect(self, first_token_time: float, last_token_time: float) -> float:
        if first_token_time is None or last_token_time is None:
            return 0.0
        progress = last_token_time - first_token_time
        return self.delta * (progress ** 0.7)

    def _complexity_penalty(self, request_complexity: float) -> float:
        if request_complexity is None:
            return 0.0
        return self.epsilon * (request_complexity ** 0.5)

    def _deadline_bonus(self, current_time: float, deadline: float, priority: int) -> float:
        if deadline is None:
            return 0.0
        time_left = deadline - current_time
        if time_left <= 0:
            return -100.0  
        base_score = 10 / (1 + time_left**0.5) 
        return base_score * (1.5 if priority > 0 else 1.0)

    def _resource_penalty(self, resource_requirements: dict) -> float:
        if resource_requirements is None:
            return 0.0
        _memory = resource_requirements.get("_memory", 0.0)
        cpu_cores = resource_requirements.get("cpu_cores", 0.0)
        return -0.5 * (_memory + 0.2 * cpu_cores) 

    def _softmax_scaling(self, scores: list) -> list:
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        sum_exp_scores = sum(exp_scores)
        return [score / sum_exp_scores for score in exp_scores]


    def total_score(
        self,
        current_time: float,
        task: Task,
    ) -> float:
        metadata = task.metadata
        arrival_priority = self._arrival_priority(current_time, metadata.arrival_time, metadata.priority)
        queue_effect = self._queue_time_effect(metadata.time_in_queue, metadata.priority)
        schedule_bonus = self._schedule_bonus(metadata.first_scheduled_time)
        progress_effect = self._execution_progress_effect(metadata.first_token_time, metadata.last_token_time)
        complexity_penalty = self._complexity_penalty(metadata.request_complexity)
        deadline_bonus = self._deadline_bonus(current_time, metadata.deadline, metadata.priority)
        resource_penalty = self._resource_penalty(metadata.resource_requirements)
        factors = [arrival_priority, queue_effect, schedule_bonus, progress_effect, deadline_bonus]
        normalized_factors = self._softmax_scaling(factors)
        score = (
            normalized_factors[0] * arrival_priority +
            normalized_factors[1] * queue_effect +
            normalized_factors[2] * schedule_bonus +
            normalized_factors[3] * progress_effect +
            normalized_factors[4] * deadline_bonus -
            complexity_penalty +
            resource_penalty
        )
        return score

class Profilers:

    _PROFILERS = {
        'fcfs': Profiler_FCFS,
        'tas_v1': Profiler_TAS_v1,
        'tas_v2': Profiler_TAS_v2,
        'tas_v3': Profiler_TAS_v3,
        'tas_v4': Profiler_TAS_v4,
        'tas_v5': Profiler_TAS_v5,
        }

    @classmethod
    def get_profiler(cls, _name: str, **kwargs) -> Profiler:
        return cls._PROFILERS[_name](**kwargs)
