import torch
import sinfer
from sinfer.logger import init_logger
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from sinfer.model_executor.layers.activation import SiluAndMul, GeluAndMul, NewGELU, FastGELU

logger = init_logger(__name__)


def sinfer__model_executor__layers__activation__SiluAndMul__forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward(x)


def sinfer__model_executor__layers__activation__GeluAndMul__forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward(x)


def sinfer__model_executor__layers__activation__NewGELU__forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward(x)


def sinfer__model_executor__layers__activation__FastGELU__forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward(x)


MluHijackObject.add_hijack_object(SiluAndMul,
                                  SiluAndMul.forward,
                                  sinfer__model_executor__layers__activation__SiluAndMul__forward)
MluHijackObject.add_hijack_object(GeluAndMul,
                                  GeluAndMul.forward,
                                  sinfer__model_executor__layers__activation__GeluAndMul__forward)
MluHijackObject.add_hijack_object(NewGELU,
                                  NewGELU.forward,
                                  sinfer__model_executor__layers__activation__NewGELU__forward)
MluHijackObject.add_hijack_object(FastGELU,
                                  FastGELU.forward,
                                  sinfer__model_executor__layers__activation__FastGELU__forward)