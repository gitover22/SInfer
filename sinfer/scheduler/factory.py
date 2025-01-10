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

        time_in_queue = metadata.time_in_pool or 0.0

        first_scheduled_ = 1.0 if metadata.first_scheduled_time else 0.0

        execution_progress = (metadata.last_token_time or 0.0) - (metadata.first_token_time or 0.0)

        length_score = self.promptLen_score(metadata.prompt_length)

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
        wait_time = max(current_time - arrival_time, 0)  # 确保非负
        return self.alpha * (wait_time ** 0.5)

    def _queue_time_effect(self, time_in_queue: float) -> float:
        time_in_queue = max(time_in_queue, 0)  # 确保非负
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
            progress = max(last_token_time - first_token_time, 0)  # 确保非负
            return self.delta * (progress ** 0.7)
        return 0.0

    def _complex_weighting(self, arrival_priority: float, queue_effect: float, penalty: float, progress_effect: float) -> float:
        # 检查输入是否包含复数
        if any(isinstance(x, complex) for x in [arrival_priority, queue_effect, penalty, progress_effect]):
            raise ValueError(f"Complex number detected in weighting inputs: {arrival_priority}, {queue_effect}, {penalty}, {progress_effect}")
        
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
        queue_effect = self._queue_time_effect(metadata.time_in_pool or 0.0)
        schedule_penalty = self._schedule_penalty(metadata.first_scheduled_time or 0.0)
        progress_effect = self._execution_progress_effect(metadata.first_token_time or 0.0, metadata.last_token_time or 0.0)

        score = self._complex_weighting(arrival_priority, queue_effect, schedule_penalty, progress_effect)

        return score

class Profiler_TAS_v3(Profiler):
    def __init__(self, alpha: float = 1.0, alpha2: float = 3.0, 
                 beta: float = 0.1,scale:int =100, degree: int = 3):
        self.alpha = alpha # a1
        self.alpha2 = alpha2 # a2
        self.beta = beta # b
        self.scale = scale
        self.degree = degree

    def _polynomial_sum(self, x: float, degree: int = 1) -> float:
        return sum([(math.pow(x, i)) / math.factorial(i) for i in range(1, degree + 1)])

    def _arrival_score(self, current_time: float, arrival_time: float, priority: int) -> float:
        if arrival_time is None:
            return 0.0
        wait_time = current_time - arrival_time
        poly_sum = self._polynomial_sum(wait_time, self.degree)
        priority_factor = math.exp(0.2 * priority)
        return self.alpha * poly_sum * priority_factor

    def time_in_queue_score(self, time_in_queue: float, priority: int) -> float:
        if time_in_queue is None or time_in_queue == 0.0:
            return 0.0
        poly_sum = self._polynomial_sum(time_in_queue, self.degree)
        priority_factor = math.exp(priority)
        return self.alpha2 * poly_sum * priority_factor
    def _schedule_score(self, first_scheduled_time: float) -> float:
        if first_scheduled_time is None:
            return 0.0
        else:
            schedule_scale = math.log(self.scale)
        poly_sum = self._polynomial_sum(schedule_scale)
        bonus = math.log(poly_sum)
        return self.beta * bonus

    def promptLen_score(self, promptLen: int) -> float:
        if promptLen is None or promptLen == 0:
            return 0.0
        poly_sum = self._polynomial_sum(promptLen/self.scale, self.degree)
        if poly_sum < 1:
            penalty = math.log(poly_sum + 1)
        else:
            penalty = math.log(poly_sum)
        return self.beta*penalty

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
            wait_score = self._arrival_score(current_time, arrival_time, priority)
            score += wait_score
        else:
            metadata.arrival_time = current_time

        if metadata.time_in_pool is not None:
            time_in_queue = metadata.time_in_pool
            queue_score = self.time_in_queue_score(time_in_queue, priority)
            score += queue_score
        else:
            metadata.time_in_pool = 0.0

        if metadata.first_scheduled_time is not None:
            schedule_score = self._schedule_score(metadata.first_scheduled_time)
            score += schedule_score

        if metadata.prompt_length is not None:
            length_score = self.promptLen_score(metadata.prompt_length)
            score -= length_score
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
        if metadata.time_in_pool is not None:
            time_in_queue = metadata.time_in_pool
            queue_score = self.time_in_queue_score(time_in_queue, priority)
            score += queue_score
        else:
            metadata.time_in_pool = 0.0
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

    def _validate_real(self, value: float) -> float:
        """Ensure the value is a real number and not a complex number."""
        if isinstance(value, complex):
            return float(value.real)  # Return the real part if it's complex
        return float(value)

    def _arrival_priority(self, current_time: float, arrival_time: float, priority: int) -> float:
        if arrival_time is None:
            return 0.0
        wait_time = current_time - arrival_time
        base_score = self.alpha * (1 / (1 + math.exp(-0.1 * wait_time))) 
        return self._validate_real(base_score * (1.5 if priority > 0 else 1.0))

    def _queue_time_effect(self, time_in_queue: float, priority: int) -> float:
        if time_in_queue is None or time_in_queue == 0.0:
            return 0.0
        base_score = self.beta * (2 * (time_in_queue ** 0.5))
        return self._validate_real(base_score * (1.2 if priority > 0 else 1.0))

    def _schedule_bonus(self, first_scheduled_time: float) -> float:
        if first_scheduled_time is None:
            return 0.0
        return self._validate_real(self.gamma * (1 / (1 + first_scheduled_time)))

    def _execution_progress_effect(self, first_token_time: float, last_token_time: float) -> float:
        if first_token_time is None or last_token_time is None:
            return 0.0
        progress = last_token_time - first_token_time
        return self._validate_real(self.delta * (progress ** 0.7))

    def _complexity_penalty(self, request_complexity: float) -> float:
        if request_complexity is None:
            return 0.0
        return self._validate_real(self.epsilon * (request_complexity ** 0.5))

    def _deadline_bonus(self, current_time: float, deadline: float, priority: int) -> float:
        if deadline is None:
            return 0.0
        time_left = deadline - current_time
        if time_left <= 0:
            return -100.0  
        base_score = 10 / (1 + time_left**0.5) 
        return self._validate_real(base_score * (1.5 if priority > 0 else 1.0))

    def _resource_penalty(self, resource_requirements: dict) -> float:
        if resource_requirements is None:
            return 0.0
        _memory = resource_requirements.get("_memory", 0.0)
        cpu_cores = resource_requirements.get("cpu_cores", 0.0)
        return self._validate_real(-0.5 * (_memory + 0.2 * cpu_cores))

    def _softmax_scaling(self, scores: list) -> list:
        max_score = max(scores)
        exp_scores = [math.exp(self._validate_real(score - max_score)) for score in scores]
        sum_exp_scores = sum(exp_scores)
        return [self._validate_real(score / sum_exp_scores) for score in exp_scores]

    def total_score(
        self,
        current_time: float,
        task: Task,
    ) -> float:
        metadata = task.metadata
        arrival_priority = self._arrival_priority(current_time, metadata.arrival_time, metadata.priority)
        queue_effect = self._queue_time_effect(metadata.time_in_pool, metadata.priority)
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
        return self._validate_real(score)

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
