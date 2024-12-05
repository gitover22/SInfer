from typing import Optional, Union, Tuple
import torch
import sinfer
from sinfer.logger import init_logger
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from bangtransformer.torch import bt_ops
from sinfer.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)


def sinfer__model_executor__layers__layernorm__RMSNorm__forward(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: use bt_ops.fused_rms_norm
    '''
    x = x.view(-1, self.weight.data.shape[0])
    if residual is not None:
        residual = residual.view(-1, self.weight.data.shape[0])
        return bt_ops.fused_rms_norm(x, residual, self.weight.data, None, None, self.variance_epsilon, True)
    else:
        return bt_ops.fused_rms_norm(x, residual, self.weight.data, None, None, self.variance_epsilon, False)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


MluHijackObject.add_hijack_object(RMSNorm,
                                  RMSNorm.forward,
                                  sinfer__model_executor__layers__layernorm__RMSNorm__forward)