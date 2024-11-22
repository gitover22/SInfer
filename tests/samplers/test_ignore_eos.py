"""Make sure ignore_eos works.

Run `pytest tests/samplers/test_ignore_eos.py`.
"""

import pytest

from sinfer import SamplingParams

import os

MODELS = [os.environ["TEST_MODEL_PATH"]]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [1024])
def test_beam_search_single_input(
    sinfer_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    example_prompts = "1 + 1 is"

    sinfer_model = sinfer_runner(model, dtype=dtype)
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    ignore_eos_output = sinfer_model.model.generate(
        example_prompts, sampling_params=sampling_params)
    print(len(ignore_eos_output[0].outputs[0].token_ids))
    assert max_tokens - len(ignore_eos_output[0].outputs[0].token_ids) < 10
    assert max_tokens - len(ignore_eos_output[0].outputs[0].token_ids) >= 0
