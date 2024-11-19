"""Compare the outputs of HF and SInfer when using greedy sampling.

This tests bigger models and use half precision.

Run `pytest tests/models/test_big_models.py`.
"""
import pytest

import os

MODELS = [
    # "meta-llama/Llama-2-7b-hf",
    # # "mistralai/Mistral-7B-v0.1",  # Broken
    # # "Deci/DeciLM-7b",  # Broken
    # # "tiiuae/falcon-7b",  # Broken
    # "EleutherAI/gpt-j-6b",
    # "mosaicml/mpt-7b",
    # # "Qwen/Qwen1.5-0.5B"  # Broken,
    os.environ["TEST_MODEL_PATH"]
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
def test_models(
    hf_runner,
    sinfer_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    sinfer_model = sinfer_runner(model, dtype=dtype)
    sinfer_outputs = sinfer_model.generate_greedy(example_prompts, max_tokens)
    del sinfer_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        sinfer_output_ids, sinfer_output_str = sinfer_outputs[i]
        assert hf_output_str == sinfer_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nSInfer: {sinfer_output_str!r}")
        assert hf_output_ids == sinfer_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nSInfer
            : {sinfer_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_model_print(
    sinfer_runner,
    model: str,
    dtype: str,
) -> None:
    sinfer_model = sinfer_runner(model, dtype=dtype)
    # This test is for verifying whether the model's extra_repr
    # can be printed correctly.
    print(sinfer_model.model.llm_engine.model_executor.driver_worker.
          model_runner.model)
    del sinfer_model
