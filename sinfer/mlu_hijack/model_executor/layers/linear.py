from typing import Optional
import torch
import sinfer
import torch.nn.functional as F
from sinfer.logger import init_logger
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from bangtransformer.torch import bt_ops
from sinfer.model_executor.layers.linear import UnquantizedLinearMethod

logger = init_logger(__name__)


def sinfer__model_executor__layers__linear__UnquantizedLinearMethod__apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    weight = layer.weight
    '''
    =============================
    Modify by sinfer_mlu
    =============================
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
    End of MLU Hijack
    =================
    '''
    if self.separate_bias_add:
        if bias is not None:
            return F.linear(x, weight) + bias
        return F.linear(x, weight)
    return F.linear(x, weight, bias)


MluHijackObject.add_hijack_object(UnquantizedLinearMethod,
                                  UnquantizedLinearMethod.apply,
                                  sinfer__model_executor__layers__linear__UnquantizedLinearMethod__apply)