from typing import Optional, Union, Tuple
import torch
import sinfer
from sinfer.logger import init_logger
from sinfer.Intercept.intercept_utils import InterceptObject
from bangtransformer.torch import bt_ops
from sinfer.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)


def intercept_RMSNorm_forward(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''
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
    '''


InterceptObject.add_intercept(RMSNorm,
                                  RMSNorm.forward,
                                  intercept_RMSNorm_forward)