
from sinfer.engine.arg_utils import AsyncEngineArgs, EngineArgs
from sinfer.engine.async_llm_engine import AsyncLLMEngine
from sinfer.engine.llm_engine import LLMEngine
from sinfer.entrypoints.llm import LLM
from sinfer.executor.ray_utils import initialize_ray_cluster
from sinfer.model_executor.models import ModelRegistry
from sinfer.outputs import CompletionOutput, RequestOutput
from sinfer.sampling_params import SamplingParams
import os
import warnings

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

__version__ = "1.0.0+mlu" +  find_version(os.path.dirname(os.path.abspath(__file__)) + "/version_config") + ".pt2.3"

__all__ = [
    "LLM",
    "ModelRegistry",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
]
