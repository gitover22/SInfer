import torch
import torch.nn.functional as F
from typing import Optional
from sinfer.distributed.communication_op import tensor_model_parallel_all_reduce
from bangtransformer.torch import bt_ops
from sinfer.logger import init_logger
from sinfer.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from sinfer.model_executor.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    ColumnParallelLinear,
    RowParallelLinear
)
from sinfer.Intercept.mlu_hijack_utils import set_is_gated

logger = init_logger(__name__)

class FeedForward(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        up_proj_name: str,
        is_gated: bool,
        down_proj_name: str,
        bias: bool,
        quant_config: Optional[QuantizationConfig] = None,
        skip_bias_add: bool = False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.reduce_results = reduce_results
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.is_gated = is_gated
        self.bias = bias
        self.up_proj_name = up_proj_name
        self.down_proj_name = down_proj_name
        self.quant_config = quant_config
        self.is_initialized = False
        self.skip_bias_add = skip_bias_add
        self.reduce_results = reduce_results
        self.use_bt_ffn = True if quant_config is None else False
        set_is_gated(self.is_gated)

        # up_proj with gate or not
        if self.is_gated:
            up_proj = MergedColumnParallelLinear(hidden_size,
                                                 [intermediate_size] * 2,
                                                 bias=bias,
                                                 quant_config=quant_config)
        else:
            up_proj = ColumnParallelLinear(hidden_size,
                                           intermediate_size,
                                           bias=bias,
                                           skip_bias_add=skip_bias_add,
                                           quant_config=quant_config)
        self.register_module(up_proj_name, up_proj)

        # down_proj
        down_proj = RowParallelLinear(intermediate_size,
                                      hidden_size,
                                      bias=bias,
                                      skip_bias_add=skip_bias_add,
                                      reduce_results=reduce_results,
                                      quant_config=quant_config)
        self.register_module(down_proj_name, down_proj)

    def prepare_weight(self):
        if not self.is_initialized:
            # alpha and beta are 1.0 and 0.0 respectively due to the fact that we don't need residual for now
            self.alpha = 1.0
            self.beta = 0.0
            # place it here to avoid the overhead of calling it in the forward pass
            self.is_initialized = True

    def _forward(self, hidden_states):
        self.prepare_weight()
        up_proj = getattr(self, self.up_proj_name)
        down_proj = getattr(self, self.down_proj_name)
        act_dict = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
        }
        fc1 = F.linear(hidden_states, up_proj.weight, bias=up_proj.bias)
        if self.is_gated:
            d = fc1.shape[-1] // 2
            fc1 = act_dict[self.hidden_act](fc1[..., :d]) * fc1[..., d:]
        else:
            fc1 = act_dict[self.hidden_act](fc1)
        fc2 = F.linear(fc1, down_proj.weight, bias=None)
        fc2 = tensor_model_parallel_all_reduce(fc2)
        if not self.skip_bias_add:
            fc2 = fc2 + down_proj.bias if down_proj.bias is not None else fc2
        return fc2

    def forward(self, hidden_states):
        self.prepare_weight()
        up_proj = getattr(self, self.up_proj_name)
        down_proj = getattr(self, self.down_proj_name)
        if self.use_bt_ffn:
            # The matmul formula is the following:
            #   mul_out = alpha * (matmul(input, filter, transpose\_b=True) + bias) + beta * residual
            #   output = active(mul_out)
            # Notes: We cannot use the activation function in matmul because it does not support gated operation
            #  we might support its in btops matmul in the future
            fc1 = bt_ops.matmul(hidden_states.view(-1, self.hidden_size), up_proj.weight, up_proj.bias, 
                                None, 'none', self.alpha, self.beta)
            act_out = bt_ops.active(fc1, self.hidden_act, self.is_gated)
            out_ = bt_ops.matmul(act_out, down_proj.weight, None, None, 'none', self.alpha, self.beta)
            # bias if existed need to add after second matmul according to the original design of sinfer
            if self.reduce_results:
                out = tensor_model_parallel_all_reduce(out_)
            else:
                out = out_
            # do the bias add if needed
            if not self.skip_bias_add:
                out = out + down_proj.bias if down_proj.bias is not None else out
            else:
                return out, down_proj.bias
        else:
            fc1, bias = up_proj(hidden_states)
            if bias is not None:
                fc1 += bias
            fc1 = bt_ops.active(fc1, self.hidden_act, self.is_gated)
            out, bias = down_proj(fc1)
            if self.skip_bias_add:
                return out, bias
        return out
