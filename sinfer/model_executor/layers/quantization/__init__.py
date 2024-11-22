from typing import Dict, Type

from sinfer.model_executor.layers.quantization.aqlm import AQLMConfig
from sinfer.model_executor.layers.quantization.awq import AWQConfig
from sinfer.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from sinfer.model_executor.layers.quantization.fp8 import Fp8Config
from sinfer.model_executor.layers.quantization.gptq import GPTQConfig
from sinfer.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from sinfer.model_executor.layers.quantization.marlin import MarlinConfig
from sinfer.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "fp8": Fp8Config,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "marlin": MarlinConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return QUANTIZATION_METHODS[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
