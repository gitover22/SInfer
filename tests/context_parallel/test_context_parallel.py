import torch
import sys
import ray
import gc
import contextlib
from sinfer import LLM, SamplingParams

def cleanup():
    """Release occupied resources and reset parallel_state"""
    from sinfer.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()

def run_sinfer(prompts, sampling_params, tp, cp):
    """Run LLM"""
    llm = LLM(model="/data/AE/llm/models/Llama-2-7b-hf/",
              enforce_eager=True,
              tensor_parallel_size = tp,
              context_parallel_size = cp)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def test_context_parallel():
    """Compare the output results of cp1 and cp2"""
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, max_tokens=16)
    outputs_1 = run_sinfer(prompts, sampling_params, tp=1, cp=1)
    cleanup()
    outputs_2 = run_sinfer(prompts, sampling_params, tp=1, cp=2)
    cleanup()
    generated_text_1 = [output.outputs[0].text for output in outputs_1]
    generated_text_2 = [output.outputs[0].text for output in outputs_2]
    assert generated_text_1 == generated_text_2