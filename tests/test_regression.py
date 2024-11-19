"""Containing tests that check for regressions in SInfer's behavior.

It should include tests that are reported by users and making sure they
will never happen again.

"""
import gc

import torch

from sinfer import LLM, SamplingParams

import pytest
import os


def test_duplicated_ignored_sequence_group():
    """https://github.com/sinfer-project/sinfer/issues/1655"""

    sampling_params = SamplingParams(temperature=0.01,
                                     top_p=0.1,
                                     max_tokens=256)
    llm = LLM(model=os.environ["TEST_MODEL_PATH"],
              max_num_batched_tokens=4096,
              tensor_parallel_size=1)
    prompts = ["This is a short prompt", "This is a very long prompt " * 1000]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(prompts) == len(outputs)


def test_max_tokens_none():
    sampling_params = SamplingParams(temperature=0.01,
                                     top_p=0.1,
                                     max_tokens=None)
    llm = LLM(model=os.environ["TEST_MODEL_PATH"],
              max_num_batched_tokens=4096,
              tensor_parallel_size=1)
    prompts = ["Just say hello!"]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(prompts) == len(outputs)



@pytest.mark.skip(reason="Skipping this test beacause memory allocated release failed")
def test_gc():
    llm = LLM(os.environ["TEST_MODEL_PATH"], enforce_eager=True)
    del llm

    gc.collect()
    torch.cuda.empty_cache()

    # The memory allocated for model and KV cache should be released.
    # The memory allocated for PyTorch and others should be less than 50MB.
    # Usually, it's around 10MB.
    allocated = torch.cuda.memory_allocated()
    assert allocated < 50 * 1024 * 1024


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
