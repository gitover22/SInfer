from sinfer import LLM, SamplingParams
from sinfer.transformers_utils.config import get_config
from sinfer.model_executor.layers.quantization import QUANTIZATION_METHODS
import argparse
import numpy as np
import time
import torch
from tqdm import tqdm
from typing import Optional

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        help="support /data/AE/llm/models/Llama-2-7b-hf/, \
                            /data/AE/llm/models/Llama-2-13b-hf/, \
                            /data/AE/llm/models/Llama-2-70b-hf/")
    parser.add_argument('--input_len', type=int, default=4096)
    parser.add_argument('--output_len', type=int, default=1)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, help="tp")
    parser.add_argument("--context_parallel_size", "-cp", type=int, help="cp")
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument('--num_iters_warmup',
                        type=int,
                        default=3,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num_iters',
                        type=int,
                        default=10,
                        help='Number of iterations to run.')
    parser.add_argument('--trust_remote_code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--latency',
                        action='store_true',
                        help='get context latency')
    args = parser.parse_args()
    
    print("model: ", args.model)
    print("seq_len: ", args.input_len)
    print("tensor_parallel_size: ", args.tensor_parallel_size)
    print("context_parallel_size: ", args.context_parallel_size)
    
    sampling_params = SamplingParams(temperature=0.8, max_tokens=args.output_len)
    llm = LLM(model=args.model, enforce_eager=True, max_model_len = args.input_len,
              max_num_batched_tokens = args.input_len, max_num_seqs = 1,
              tensor_parallel_size = args.tensor_parallel_size,
              context_parallel_size = args.context_parallel_size)
    
    np.random.seed(0)    
    dummy_prompt_token_ids = np.random.randint(10000, size=(1, args.input_len))
    dummy_prompt_token_ids = dummy_prompt_token_ids.tolist()
    
    if args.latency:
        def run_to_completion():
            start_time = time.perf_counter()
            llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                        sampling_params=sampling_params,
                        use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

        print("Warming up...")
        for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
            run_to_completion()

        # Benchmark.
        latencies = []
        for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
            latencies.append(run_to_completion())
        latencies = np.array(latencies)
        percentages = [10, 25, 50, 75, 90]
        percentiles = np.percentile(latencies, percentages)
        print(f'Avg latency: {np.mean(latencies)} seconds')
        for percentage, percentile in zip(percentages, percentiles):
            print(f'{percentage}% percentile latency: {percentile} seconds')
        llm.get_metrics(args.num_iters_warmup,False,args.input_len,args.output_len,args.tensor_parallel_size,args.quantization)
    else:
        outputs = llm.generate(prompt_token_ids=dummy_prompt_token_ids, sampling_params = sampling_params)
