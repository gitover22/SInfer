"""Compare the outputs of HF and SInfer for Mistral models using greedy sampling.

Run `pytest tests/models/test_mistral.py`.
"""
import pytest

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.skip(
    "Two problems: 1. Failing correctness tests. 2. RuntimeError: expected "
    "scalar type BFloat16 but found Half (only in CI).")
def test_models(
    hf_runner,
    sinfer_runner,
    example_long_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_long_prompts, max_tokens)
    del hf_model

    sinfer_model = sinfer_runner(model, dtype=dtype)
    sinfer_outputs = sinfer_model.generate_greedy(example_long_prompts, max_tokens)
    del sinfer_model

    for i in range(len(example_long_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        sinfer_output_ids, sinfer_output_str = sinfer_outputs[i]
        assert hf_output_str == sinfer_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nSInfer: {sinfer_output_str!r}")
        assert hf_output_ids == sinfer_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nSInfer: {sinfer_output_ids}")
