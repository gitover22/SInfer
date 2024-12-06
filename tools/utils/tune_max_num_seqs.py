"""Autotune max_num_seqs paramter."""
# pylint: skip-file
import argparse
import random
from typing import Dict, Any
from tqdm import tqdm


def run_sinfer(config: Dict[str, Any]) -> float:
    """Initialize and run an instance of a language model (LLM) using the
    `sinfer` library."""
    print(f'Evaluate with max_num_seqs: {config["max_num_seqs"]}')
    from sinfer import LLM
    llm = LLM(**config)
    print(f'The num of gpu blocks is: {llm.llm_runtime.cache_config.num_gpu_blocks}')
    return llm.llm_runtime.cache_config.num_gpu_blocks


def main(args: argparse.Namespace):
    """The entry function to tune max_num_seqs."""
    print(args)
    random.seed(args.seed)
    config = {
        'model': args.model,
        'tokenizer': args.tokenizer,
        'quantization': args.quantization,
        'tensor_parallel_size': args.tensor_parallel_size,
        'seed': args.seed,
        'trust_remote_code': args.trust_remote_code,
        'dtype': args.dtype,
        'max_model_len': args.max_model_len,
        'enforce_eager': args.enforce_eager,
        'kv_cache_dtype': args.kv_cache_dtype,
        'quantization_param_path': args.quantization_param_path,
        'device': args.device,
        'enable_prefix_caching': args.enable_prefix_caching,
        'enable_chunked_prefill': args.enable_chunked_prefill,
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'download_dir': args.download_dir,
        'block_size': args.block_size
    }

    import multiprocessing
    def worker_wrapper(config, output_queue):
        """Here we get the num_gpu_blocks by instantiate a llm object."""
        result = run_sinfer(config)
        output_queue.put(result)


    def get_num_gpu_blocks(cache, num_seqs) -> int:
        """Get the number of GPU blocks with parameter num_seqs."""
        if num_seqs in cache:
            return cache[num_seqs]
        # Here since we cannot manually release the resources hold by Ray and NCCL,
        # we evaluate a set of parameters by launching a separate process.
        config['max_num_seqs'] = num_seqs
        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=worker_wrapper,
                                          args=(config, output_queue))
        process.start()
        process.join()
        result = output_queue.get()
        cache[num_seqs] = result
        return result


    def find_optimal_max_num_seqs(init=256) -> int:
        """Search th optimal max_num_seqs which maximizes
        min(max_num_seqs, num_gpu_blocks)."""
        # Use cache to avoid repeated evaluations.
        cache = {}

        # Initialization seach range.
        num_blocks = get_num_gpu_blocks(cache, init)
        left, right = min(num_blocks, init), max(num_blocks, init)

        # Binary search.
        while 0 < left < right:
            mid = (left + right) // 2
            num_blocks = get_num_gpu_blocks(cache, mid)

            if num_blocks == mid:
                return mid
            if num_blocks > mid:
                left = mid + 1
            else:
                right = mid - 1
            left = max(min(mid, num_blocks), left)
            right = min(max(mid, num_blocks), right)

        left, right = max(1, left), max(1, right)
        final_left = min(left, get_num_gpu_blocks(cache, left))
        final_right = min(right, get_num_gpu_blocks(cache, right))
        return right if final_right > final_left else left

    max_num_seqs = find_optimal_max_num_seqs()
    print(f'The optimal max_num_seqs is {max_num_seqs}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune max_num_seqs.")
    parser.add_argument("--backend", type=str, choices=["sinfer"], default="sinfer")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len", type=int, default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization', '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int, default=None,
                        help="Maximum batch size for HF backend.")

    parser.add_argument("--block-size", type=int, default=-1)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len', type=int, default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype', type=str, default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager", action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype", type=str, choices=["auto", "fp8"], default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        '--quantization-param-path', type=str, default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda"],
        help='device type for SInfer execution, supporting CUDA only currently.')
    parser.add_argument(
        "--enable-prefix-caching", action='store_true',
        help="enable automatic prefix caching for SInfer backend.")
    parser.add_argument("--enable-chunked-prefill", action='store_true',
                        help="enable chunked prefill for SInfer backend.")
    parser.add_argument('--max-num-batched-tokens', type=int, default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir', type=str, default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    cli_args = parser.parse_args()
    if cli_args.tokenizer is None:
        cli_args.tokenizer = cli_args.model
    if cli_args.dataset is None:
        assert cli_args.input_len is not None
        assert cli_args.output_len is not None
    else:
        assert cli_args.input_len is None

    main(cli_args)
