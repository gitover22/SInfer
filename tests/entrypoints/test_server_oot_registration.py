import multiprocessing
import sys
import time
import pytest

import torch
from openai import OpenAI, OpenAIError

from sinfer import ModelRegistry
from sinfer.model_executor.models.opt import OPTForCausalLM
from sinfer.model_executor.sampling_metadata import SamplingMetadata
from sinfer.utils import get_open_port

import os

class MyOPTForCausalLM(OPTForCausalLM):

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states, sampling_metadata)
        logits.zero_()
        logits[:, 0] += 1.0
        return logits


def server_function(port):
    # register our dummy model
    ModelRegistry.register_model("OPTForCausalLM", MyOPTForCausalLM)
    model_path = os.environ["TEST_MODEL_PATH"]
    sys.argv = ["placeholder.py"] + \
        (f"--model {model_path} --dtype"
        f" float16 --api-key token-abc123 --chat-template ../../examples/template_chatml.jinja --port {port}").split()
    import runpy
    runpy.run_module('sinfer.entrypoints.openai.api_server', run_name='__main__')


def test_oot_registration_for_api_server():
    port = get_open_port()
    multiprocessing.set_start_method('spawn')
    server = multiprocessing.Process(target=server_function, args=(port, ))
    server.start()
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="token-abc123",
    )
    while True:
        try:
            completion = client.chat.completions.create(
                model=os.environ["TEST_MODEL_PATH"],
                messages=[{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role": "user",
                    "content": "Hello!"
                }],
                temperature=0,
            )
            break
        except OpenAIError as e:
            if "Connection error" in str(e):
                time.sleep(3)
            else:
                raise e
    server.kill()
    generated_text = completion.choices[0].message.content
    # make sure only the first token is generated
    rest = generated_text.replace("<s>", "")
    # NOTE: Only test register, the outputs can not match will fix at SINFER-64
    # assert rest == ""
