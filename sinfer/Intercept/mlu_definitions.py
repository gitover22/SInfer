import torch
from typing import Dict, List, Tuple
import os

LATENCY_DEBUG           = "SINFER_LATENCY_DEBUG"
LATENCY_DEBUG_NO_DEVICE = "SINFER_LATENCY_DEBUG_NO_DEVICE"
DUMP_CPU_INFO           = "SINFER_DUMP_CPU_INFO"
DUMP_MLU_INFO           = "SINFER_DUMP_MLU_INFO"

USE_PAGED = "3" not in torch.cuda.get_device_name(0)
if 'USE_PAGED' in os.environ:
    if os.environ['USE_PAGED'] in ("True","true", "TRUE",'1'):
        USE_PAGED = True
    else:
        USE_PAGED = False
if USE_PAGED:
    print("\n============================USE PAGED=============================\n")
else:
    print("\n============================NO PAGED=============================\n")

USE_CONTEXT_PARALLEL = 'USE_CONTEXT_PARALLEL'
SCHEDULER_PROFILE_ENABLE = 'SINFER_SCHEDULER_PROFILE_ENABLE'
CUDA_GRAPH_DEBUG = os.environ["SINFER_GRAPH_DEBUG"].lower() in ["true", "1"] \
                    if "SINFER_GRAPH_DEBUG" in os.environ else True

# 设置块大小
class BlockSizeInfo :
    BLOCK_SIZE = -1

    @classmethod
    def set_block_size(cls, a : int) :
        if USE_PAGED :
            if a != -1 and a != 16 :
                raise ValueError("BLOCKSIZE other than 16 are not supported in paged mode, please check '--block-size' value.")
            cls.BLOCK_SIZE = 16
        else :
            cls.BLOCK_SIZE = 1024 if a == -1 else a   # 说明：必须满足BLOCK_SIZE > input_len + max_tokens （BLOCK_SIZE和模型最大文本长度max_model_len无关）

# 根据环境变量的值，动态决定某些功能是否开启或执行
def _check_env_true(env):
    return os.environ[env].lower() in ["true", "1"] if env in os.environ else False

def check_latency_debug():
    return _check_env_true(LATENCY_DEBUG_NO_DEVICE) or _check_env_true(LATENCY_DEBUG)

def check_latency_debug_with_device():
    return _check_env_true(LATENCY_DEBUG) and not _check_env_true(LATENCY_DEBUG_NO_DEVICE)

def check_context_parallel():
    return _check_env_true(USE_CONTEXT_PARALLEL)

def check_scheduler_profile_enable():
    return _check_env_true(SCHEDULER_PROFILE_ENABLE)

def check_dump_cpu_info():
    return check_latency_debug_with_device() and  _check_env_true(DUMP_CPU_INFO)

def check_dump_mlu_info():
    return check_latency_debug_with_device() and _check_env_true(DUMP_MLU_INFO)
