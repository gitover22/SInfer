
from typing import Type
import torch
import sinfer
import sinfer.envs as envs
from sinfer.logger import init_logger
from sinfer.utils import is_cpu
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.attention.selector import _Backend

logger = init_logger(__name__)


def intercept_which_attn_to_use(dtype: torch.dtype) -> _Backend:
    """返回应该使用哪个 Flash Attention 后端."""
    if is_cpu():
        return _Backend.TORCH_SDPA
    # 增加mlu设备上的后端
    if 3 <= torch.mlu.get_device_capability()[0] <= 5 and dtype in (torch.float16, torch.bfloat16):
        logger.info("Use mlu flash_atten as Attention backend.")
        return _Backend.FLASH_ATTN

    # N 卡.
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta & Turing
        logger.info("Cannot use FlashAttention-2 backend for Volta and Turing "
                    "GPUs.")
        return _Backend.XFORMERS

    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashAttention-2 backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
        return _Backend.XFORMERS

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info(
            "Cannot use FlashAttention-2 backend because the flash_attn "
            "package is not found. Please install it for better performance.")
        return _Backend.XFORMERS

    backend_by_env_var = envs.SINFER_ATTENTION_BACKEND
    if backend_by_env_var is not None:
        return _Backend[backend_by_env_var]

    # 默认
    return _Backend.FLASH_ATTN


InterceptObject.add_intercept(sinfer.attention.selector,
                                  sinfer.attention.selector._which_attn_to_use,
                                  intercept_which_attn_to_use)