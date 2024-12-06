from typing import Optional
import torch
import sinfer
import torch.nn.functional as F
from sinfer.logger import init_logger
from sinfer.Intercept.intercept_utils import InterceptObject
from bangtransformer.torch import bt_ops
from sinfer.model_executor.layers.linear import UnquantizedLinearMethod

logger = init_logger(__name__)


def intercept_UnquantizedLinearMethod_apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    weight = layer.weight
    '''
    @brief: use bt_ops.matmul
    '''
    if weight.shape[0] >= 16:
        if self.separate_bias_add:
            if bias:
                return bt_ops.matmul(x, weight, None, None, "none", 1.0, 0) + bias
            return bt_ops.matmul(x, weight, None, None, "none", 1.0, 0)
        return bt_ops.matmul(x, weight, bias, None, "none", 1.0, 0)
    '''
    =================
    '''
    if self.separate_bias_add:
        if bias is not None:
            return F.linear(x, weight) + bias
        return F.linear(x, weight)
    return F.linear(x, weight, bias)


InterceptObject.add_intercept(UnquantizedLinearMethod,
                                  UnquantizedLinearMethod.apply,
                                  intercept_UnquantizedLinearMethod_apply)