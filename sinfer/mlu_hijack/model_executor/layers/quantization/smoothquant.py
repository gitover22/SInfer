from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter
from sinfer.model_executor.layers.linear import (LinearMethodBase, LinearBase,
                                               set_weight_attrs)
from sinfer.model_executor.layers.quantization import QUANTIZATION_METHODS
from sinfer.model_executor.layers.quantization.base_config import QuantizationConfig
from torch.nn import functional as F

from bangtransformer.torch import bt_ops

from sinfer.logger import init_logger
logger = init_logger(__name__)


class SmoothQuantConfig(QuantizationConfig):
    """Config class for SmoothQuant.
    """

    def __init__(
        self,
        weight_bits: int,
        quant_mode: str,  # smoothquant
        input_quant_method: str, # per token/per tensor
    ) -> None:
        self.weight_bits = weight_bits
        self.quant_mode = quant_mode
        self.input_quant_method = input_quant_method

        if quant_mode == "SmoothQuant" and (self.weight_bits != 8):
            raise ValueError(
                "Currently, only 8-bit weight quantization is supported for "
                f"SmoothQuant, but got {self.weight_bits} bits.")
        if quant_mode == "SmoothQuant" and (self.input_quant_method != "per_token" and self.input_quant_method != "per_tensor"):
            raise ValueError(
                "Currently, only per_token or per_tensor input quantization is supported for "
                f"SmoothQuant, but got {self.input_quant_method}.")
        self.pack_factor = 8 // self.weight_bits

    def __repr__(self) -> str:
        return (f"SmoothQuantConfig(weight_bits={self.weight_bits}, "
                f"input_quant_method={self.input_quant_method}, "
                f"quant_mode={self.quant_mode})")

    def get_name(self) -> str:
        return "SmoothQuant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_min_capability(self) -> int:
        return 30

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothQuantConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        input_quant_method = cls.get_from_keys(config, ["input_quant_method"])
        quant_mode = cls.get_from_keys(config, ["quant_mode"])
        return cls(weight_bits, quant_mode, input_quant_method)

    def get_quant_method(self, layer: torch.nn.Module) -> Optional["SmoothQuantLinearMethod"]:
        if isinstance(layer, LinearBase):
            return SmoothQuantLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class SmoothQuantLinearMethod(LinearMethodBase):
    """Linear method for SmoothQuant.

    Args:
        quant_config: The SmoothQuant quantization config.
    """

    def __init__(self, quant_config: SmoothQuantConfig):
        self.quant_config = quant_config

    def create_weights(
        self, layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int], input_size: int,
        output_size: int, params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> Dict[str, Any]:
        output_size_per_partition = sum(output_partition_sizes)
        if self.quant_config.quant_mode == "SmoothQuant":
            input_dim = None
            if input_size != input_size_per_partition:
                input_dim = 0
            qweight = Parameter(
                torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // self.quant_config.pack_factor,
                    device="cuda",
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            set_weight_attrs(
                qweight, {
                    "input_dim": 1,
                    "output_dim": 0,
                })
            per_channel_scale = Parameter(
                torch.empty(
                    output_size_per_partition,
                    device="cuda",
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            set_weight_attrs(per_channel_scale, {
                "input_dim": None,
                "output_dim": 0,
            })
            layer.register_parameter("qweight", qweight)
            set_weight_attrs(qweight, extra_weight_attrs)
            layer.register_parameter("per_channel_scale", per_channel_scale)
            set_weight_attrs(per_channel_scale, extra_weight_attrs)
            if self.quant_config.input_quant_method == "per_token":
                smooth = Parameter(
                    torch.empty(
                        input_size_per_partition,
                        device="cuda",
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                set_weight_attrs(smooth, {
                    "input_dim": input_dim,
                    "output_dim": None,
                    "ignore_warning": True,
                })
                layer.register_parameter("smooth", smooth)
                set_weight_attrs(smooth, extra_weight_attrs)
            if self.quant_config.input_quant_method == "per_tensor":
                scale_to_int = Parameter(
                    torch.empty(
                        input_size_per_partition,
                        device="cuda",
                        dtype=params_dtype,
                    ),
                    requires_grad=False,
                )
                set_weight_attrs(scale_to_int, {
                    "input_dim": input_dim,
                    "output_dim": None,
                    "ignore_warning": True,
                })
                layer.register_parameter("scale_to_int", scale_to_int)
                set_weight_attrs(scale_to_int, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.quant_config.input_quant_method == "per_token":
            quant_input, input_scale = bt_ops.per_token_smooth_quantize(x, layer.smooth, None)
        if self.quant_config.input_quant_method == "per_tensor":
            quant_input = bt_ops.quantize(x, layer.scale_to_int.to(torch.float), None)
            input_scale=None
        out = bt_ops.smooth_quant_matmul(quant_input, input_scale, layer.qweight, layer.per_channel_scale.to(torch.float), x.dtype, bias)
        return out


# add smoothquant to QUANTIZATION_METHODS
QUANTIZATION_METHODS["smoothquant"] = SmoothQuantConfig
