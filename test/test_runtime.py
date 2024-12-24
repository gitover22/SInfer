import argparse
from typing import List, Tuple

from sinfer import RuntimeArgs, LLMRuntime, RequestOutput, SamplingParams


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2,
                        best_of=5,
                        temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(n=3, best_of=3, use_beam_search=True,
                        temperature=0.0)),
    ]


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
    test_prompts = create_test_prompts()
    process_requests(runtime, test_prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMRuntime class directly')
    parser = RuntimeArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
