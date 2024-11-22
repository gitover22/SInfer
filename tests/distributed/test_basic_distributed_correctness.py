"""Compare the outputs of HF and distributed SInfer when using greedy sampling.
SInfer will allocate all the available memory, so we need to run the tests one
by one. The solution is to pass arguments (model name) by environment
variables.
Run:
```sh
TEST_DIST_MODEL=facebook/opt-125m pytest \
    test_basic_distributed_correctness.py
TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf \
    test_basic_distributed_correctness.py
```
"""
import os

import pytest
import torch

MODELS = [
    os.environ["TEST_MODEL_PATH"],
]
SINFER_ATTENTION_BACKEND = "SINFER_ATTENTION_BACKEND"


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
    hf_runner,
    sinfer_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    enforce_eager = False
    backend_by_env_var = os.getenv(SINFER_ATTENTION_BACKEND)
    if backend_by_env_var == "FLASHINFER":
        enforce_eager = True

    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    sinfer_model = sinfer_runner(model,
                             dtype=dtype,
                             tensor_parallel_size=2,
                             enforce_eager=enforce_eager)
    sinfer_outputs = sinfer_model.generate_greedy(example_prompts, max_tokens)
    del sinfer_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        sinfer_output_ids, sinfer_output_str = sinfer_outputs[i]
        assert hf_output_str == sinfer_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nSInfer: {sinfer_output_str!r}")
        assert hf_output_ids == sinfer_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nSInfer: {sinfer_output_ids}")
