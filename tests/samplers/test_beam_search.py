"""Compare the outputs of HF and SInfer when using beam search.

Run `pytest tests/samplers/test_beam_search.py`.
"""
import gc

import pytest
import torch

import os

# FIXME(zhuohan): The test can not pass if we:
#   1. Increase max_tokens to 256.
#   2. Increase beam_width to 8.
#   3. Use the model "huggyllama/llama-7b".
MAX_TOKENS = [128]
BEAM_WIDTHS = [4]
MODELS = [os.environ["TEST_MODEL_PATH"]]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
def test_beam_search_single_input(
    hf_runner,
    sinfer_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    example_prompts = example_prompts[:1]
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_beam_search(example_prompts, beam_width,
                                               max_tokens)
    del hf_model

    sinfer_model = sinfer_runner(model, dtype=dtype)
    sinfer_outputs = sinfer_model.generate_beam_search(example_prompts, beam_width,
                                                   max_tokens)
    del sinfer_model
    # NOTE(woosuk): For some reason, the following GC is required to avoid
    # GPU OOM errors in the following tests using `sinfer_runner`.
    gc.collect()
    torch.cuda.empty_cache()

    for i in range(len(example_prompts)):
        hf_output_ids, _ = hf_outputs[i]
        sinfer_output_ids, _ = sinfer_outputs[i]
        assert len(hf_output_ids) == len(sinfer_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == sinfer_output_ids[j], (
                f"Test{i} output{j}:\nHF: {hf_output_ids}\n"
                f"SInfer
                : {sinfer_output_ids}")
