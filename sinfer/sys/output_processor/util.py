from typing import List

from sinfer.request import SamplerOutput
from sinfer.task import TaskOutput


def create_output_by_task(
        sampler_outputs: List[SamplerOutput],
        num_tasks: int) -> List[List[TaskOutput]]:
    # transforms [step][task] into [task][step].
    output_by_task: List[List[SamplerOutput]] = [
        [] for _ in range(num_tasks)
    ]
    for step in sampler_outputs:
        for i, task_output in enumerate(step):
            output_by_task[i].append(task_output)

    return output_by_task
