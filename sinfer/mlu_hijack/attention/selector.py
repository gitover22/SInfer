
from typing import Type
import torch
import sinfer
import sinfer.envs as envs
from sinfer.logger import init_logger
from sinfer.utils import is_cpu, is_hip
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.attention.selector import _Backend

logger = init_logger(__name__)


def sinfer__attention__selector___which_attn_to_use(dtype: torch.dtype) -> _Backend:
    """Returns which flash attention backend to use."""
    if is_cpu():
        return _Backend.TORCH_SDPA
    # add flash_attn backend on mlu device
    # Cambricon MLUs.
    if 3 <= torch.mlu.get_device_capability()[0] <= 5 and dtype in (torch.float16, torch.bfloat16):
        logger.info("Use mlu flash_atten as Attention backend.")
        return _Backend.FLASH_ATTN
    if is_hip():
        # AMD GPUs.
        if torch.cuda.get_device_capability()[0] != 9:
            # not Instinct series GPUs.
            logger.info("flash_atten is not supported on NAVI GPUs.")
        return _Backend.ROCM_FLASH

    # NVIDIA GPUs.
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
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

    # Default case.
    return _Backend.FLASH_ATTN


MluHijackObject.add_hijack_object(sinfer.attention.selector,
                                  sinfer.attention.selector._which_attn_to_use,
                                  sinfer__attention__selector___which_attn_to_use)