import os
import sinfer
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from typing import Optional
from torch.distributed import ProcessGroup
# from sinfer.mlu_hijack.pycncl import CNCLCommunicator, cnclGetVersion
from sinfer.distributed.device_communicators import pynccl_utils
from sinfer.logger import init_logger

import sinfer.envs as envs

logger = init_logger(__name__)


def sinfer__distributed__device_communicators__pynccl_utils__init_process_group(
        group: Optional[ProcessGroup] = None) -> None:
    # 替换nccl为 CNCL
    assert not pynccl_utils.is_initialized()
    # 暂时不做这个特性
    # logger.info("SInfer is using cncl==%s", cnclGetVersion())
    # pynccl_utils.comm = CNCLCommunicator(group=group)
    # assert pynccl_utils.is_initialized()
    


MluHijackObject.add_hijack_object(pynccl_utils,
                                  pynccl_utils.init_process_group,
                                  sinfer__distributed__device_communicators__pynccl_utils__init_process_group)