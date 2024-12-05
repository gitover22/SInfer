import os
import torch
import sinfer
import sinfer.envs as envs
import numpy as np
from typing import List, Union, Optional
from sinfer.worker.model_runner import _BATCH_SIZES_TO_CAPTURE, _BATCH_SIZE_ALIGNMENT
from sinfer.logger import init_logger
import ctypes # 用于与 C/C++ 动态链接库交互

logger = init_logger(__name__)

IS_GATED=False

def set_is_gated(flag):
    global IS_GATED
    IS_GATED=flag
    
def get_is_gated():
    return IS_GATED

class MluHijackObject:
    hijack_objs = []    # 类变量,所有实例共享

    # 添加拦截对象
    @classmethod
    def add_hijack_object(cls, obj, org_func, hijack_func):
        cls.hijack_objs.append((obj, org_func, hijack_func))
    
    # 将拦截函数 (hijack_func) 替换目标对象 (obj) 的原始函数 (org_func)
    @classmethod
    def apply_hijack(cls):
        for obj, org_func, hijack_func in cls.hijack_objs:
            if type(org_func) == str:
                org_func_name = org_func
            else:
                org_func_name = org_func.__name__
            setattr(obj, org_func_name, hijack_func)
            
    # 该方法用于恢复 apply_hijack 方法所替换的原始函数 (org_func)，取消拦截
    @classmethod
    def undo_hijack(cls):
        for obj, org_func, hijack_func in cls.hijack_objs:
            if type(org_func) == str:
                delattr(obj, org_func)
            else:
                org_func_name = org_func.__name__
                setattr(obj, org_func_name, org_func)


def find_cncl_library():
    so_file = envs.SINFER_NCCL_SO_PATH
    if so_file:
        # 存在nccl库
        logger.info(
            "Found nccl from environment variable SINFER_NCCL_SO_PATH=%s",
            so_file)
    else:
        neuware_home = os.environ["NEUWARE_HOME"]
        so_file = os.path.join(neuware_home, "lib64/libcncl.so")
    return so_file

# 转化x为tensor数据类型（带填充）
def make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    padded_x = np.zeros([len(x), max_len], dtype=np.int32) + pad # 创建一个形状为 [len(x), max_len] 的 numpy 数组，默认填充值为 pad
    # 将列表的内容填充到数组对应位置
    for ind,i in enumerate(x): # ind为index，i为子列表
        padded_x[ind, :len(i)] = i # 表示对 padded_x 的第 ind 行的前 len(i) 个元素进行操作
    return torch.tensor(padded_x, dtype=dtype, device=device) # 将填充完成的 numpy 数组转换为 PyTorch 张量

# 转化为tensor数据类型（不带填充）
def make_tensor_without_pad(
    x: List[List[int]],
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
    pin_memory: bool = False,
) -> torch.Tensor:
    return torch.tensor(x,
                        dtype=dtype,
                        device=device,
                        pin_memory=pin_memory and str(device) == "cpu")

# 
def set_batch_sizes_to_capture(max_batch_size_to_capture):
    logger.info(f"Set max_batch_size_to_capture={max_batch_size_to_capture}")
    global _BATCH_SIZES_TO_CAPTURE # 一个用于存储待捕获batch size的list，升序
    if max_batch_size_to_capture <= _BATCH_SIZES_TO_CAPTURE[-1]:
        # 删除超出目标值的批量大小
        _BATCH_SIZES_TO_CAPTURE = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= max_batch_size_to_capture
        ]
    else:
        # 添加
        new_capture_batch = _BATCH_SIZES_TO_CAPTURE[-1] + _BATCH_SIZE_ALIGNMENT
        while new_capture_batch <= max_batch_size_to_capture:
            _BATCH_SIZES_TO_CAPTURE.append(new_capture_batch)
            new_capture_batch = _BATCH_SIZES_TO_CAPTURE[-1] + _BATCH_SIZE_ALIGNMENT

    if max_batch_size_to_capture not in _BATCH_SIZES_TO_CAPTURE:
        _BATCH_SIZES_TO_CAPTURE.append(max_batch_size_to_capture)



class ModelConfig(ctypes.Structure):
    _fields_ = [
        ('hidden_size', ctypes.c_double),
        ('vocab_size', ctypes.c_double),
        ('ffn_inner_size', ctypes.c_double),
        ('layer_num', ctypes.c_double),
        ('head_num', ctypes.c_double),
        ('head_size', ctypes.c_double),
        ('head_num_kv', ctypes.c_double),
        ('tp_num', ctypes.c_double),
        ('use_gated_ffn', ctypes.c_bool),
        ('experts_num', ctypes.c_int),
        ('topk_num', ctypes.c_int),
        ('use_causal_mask', ctypes.c_bool),
        ('kv_cache_dtype', ctypes.c_char_p),
        ('smooth_quant_type', ctypes.c_char_p),
        ('data_type', ctypes.c_char_p),
        ('filter_data_type', ctypes.c_char_p),
    ]

