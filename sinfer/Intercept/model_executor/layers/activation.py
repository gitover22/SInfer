import torch
import sinfer
from sinfer.logger import init_logger
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.model_executor.layers.activation import SiluAndMul, GeluAndMul, NewGELU, FastGELU

logger = init_logger(__name__)


def intercept_SiluAndMul_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward(x)


def intercept_GeluAndMul_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward(x)


def intercept_NewGELU_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward(x)


def intercept_FastGELU_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward(x)


InterceptObject.add_intercept(SiluAndMul,
                                  SiluAndMul.forward,
                                  intercept_SiluAndMul_forward)
InterceptObject.add_intercept(GeluAndMul,
                                  GeluAndMul.forward,
                                  intercept_GeluAndMul_forward)
InterceptObject.add_intercept(NewGELU,
                                  NewGELU.forward,
                                  intercept_NewGELU_forward)
InterceptObject.add_intercept(FastGELU,
                                  FastGELU.forward,
                                  intercept_FastGELU_forward)