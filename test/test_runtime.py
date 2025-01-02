import argparse
import random
from typing import List, Tuple

from sinfer import RuntimeArgs, LLMRuntime, RequestOutput, SamplingParams
from tool import read_prompts_from_csv

def create_test_prompts(args:argparse.Namespace) -> List[Tuple[str, SamplingParams]]:
    prompts = read_prompts_from_csv("prompts.csv",args.num)
    result = []
    for prompt in prompts:
        result.append((prompt, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=random.randint(16, 512))))
    return result


def process_requests(runtime: LLMRuntime,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    task_id = 0

    while test_prompts or runtime.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            runtime.add_task(str(task_id), prompt, sampling_params)
            task_id += 1

        request_outputs: List[RequestOutput] = runtime.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_runtime(args: argparse.Namespace) -> LLMRuntime:
    engine_args = RuntimeArgs.from_cli_args(args)
    return LLMRuntime.from_runtime_args(engine_args)


def main(args: argparse.Namespace):
    runtime = initialize_runtime(args)
    test_prompts = create_test_prompts(args)
    process_requests(runtime, test_prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMRuntime class directly')
    parser = RuntimeArgs.add_cli_args(parser)
    
    parser.add_argument("--num", type=int, default=8)
    args = parser.parse_args()
    args.model = './Meta-Llama-3-8B-Instruct'
    args.dtype = 'float16'
    args.tensor_parallel_size = 4
    args.gpu_memory_utilization = 1
    args.trust_remote_code=True
    main(args)
