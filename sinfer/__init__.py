
from sinfer.sys.arg_utils import AsyncEngineArgs, EngineArgs
from sinfer.sys.async_runtime import AsyncLLMEngine
from sinfer.sys.runtime import LLMRuntime
from sinfer.entrypoints.llm import LLM
from sinfer.executor.ray_utils import initialize_ray_cluster
from sinfer.model_executor.models import ModelRegistry
from sinfer.outputs import CompletionOutput, RequestOutput
from sinfer.sampling_params import SamplingParams
import os
import warnings
import subprocess
def find_version(filepath: str) -> str:
    if not os.path.exists(filepath):
        warnings.warn("Unable to find version_config file in the path {} ".format(filepath))
        return ""
    with open(filepath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if 'SINFER_VERSION=' in line:
            version = line.split('=')[1].strip()
            return version

    warnings.warn("Unable to find version string in the file {}".format(filepath))
    return ""

def get_commit_id() -> str:
    git_short_hash = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stdout=subprocess.PIPE,
        text=True
    ).stdout.strip()
    return git_short_hash

__version__ = find_version(os.path.dirname(os.path.abspath(__file__)) + "/version") +"+mlu."+get_commit_id()

__all__ = [
    "LLM",
    "ModelRegistry",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "LLMRuntime",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
]
