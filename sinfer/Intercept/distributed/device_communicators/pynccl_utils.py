import os
import sinfer
from sinfer.Intercept.intercept_utils import InterceptObject
from typing import Optional
from torch.distributed import ProcessGroup
# from sinfer.intercept.pycncl import CNCLCommunicator, cnclGetVersion
from sinfer.distributed.device_communicators import pynccl_utils
from sinfer.logger import init_logger

import sinfer.envs as envs

logger = init_logger(__name__)


def intercept_init_process_group(
        group: Optional[ProcessGroup] = None) -> None:
    # 替换nccl为 CNCL
    assert not pynccl_utils.is_initialized()
    # 暂时不做这个特性
    # logger.info("SInfer is using cncl==%s", cnclGetVersion())
    # pynccl_utils.comm = CNCLCommunicator(group=group)
    # assert pynccl_utils.is_initialized()
    


InterceptObject.add_intercept(pynccl_utils,
                                  pynccl_utils.init_process_group,
                                  intercept_init_process_group)