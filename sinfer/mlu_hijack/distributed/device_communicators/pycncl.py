# This file is a pure Python wrapper for the NCCL library.
# The main purpose is to use NCCL combined with CUDA graph.
# Before writing this script, we tried the following approach:
# 1. We tried to use `cupy`, it calls NCCL correctly, but `cupy` itself
#  often gets stuck when initializing the NCCL communicator.
# 2. We tried to use `torch.distributed`, but `torch.distributed.all_reduce`
#  contains many other potential cuda APIs, that are not allowed during
#  capturing the CUDA graph. For further details, please check
# https://discuss.pytorch.org/t/pytorch-cudagraph-with-nccl-operation-failed/ .
#
# Another rejected idea is to write a C/C++ binding for NCCL. It is usually
# doable, but we often encounter issues related with nccl versions, and need
# to switch between different versions of NCCL. See
# https://github.com/NVIDIA/nccl/issues/1234 for more details.
# A C/C++ binding is not flexible enough to handle this. It requires
# recompilation of the code every time we want to switch between different
# versions. This current implementation, with a **pure** Python wrapper, is
# more flexible. We can easily switch between different versions of NCCL by
# changing the environment variable `SINFER_NCCL_SO_PATH`, or the `so_file`
# variable in the code.

import ctypes
import platform
from typing import Optional, Union

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sinfer.distributed.parallel_state import get_cpu_world_group, get_local_rank
from sinfer.logger import init_logger
from sinfer.mlu_hijack.mlu_hijack_utils import find_cncl_library

logger = init_logger(__name__)

so_file = find_cncl_library()

try:
    # load the library in another process.
    # if it core dumps, it will not crash the current process
    cncl = ctypes.CDLL(so_file)
except Exception as e:
    logger.error(
        "Failed to load CNCL library from %s ."
        "It is expected if you are not running on CAMBRICON MLUs.",
        so_file,
        platform.platform())
    raise e

# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in


cnclResult_t = ctypes.c_int


_c_cnclGetErrorString = cncl.cnclGetErrorStr
_c_cnclGetErrorString.restype = ctypes.c_char_p
_c_cnclGetErrorString.argtypes = [cnclResult_t]


def CNCL_CHECK(result: cnclResult_t) -> None:
    if result != 0:
        error_str = _c_cnclGetErrorString(result)
        error_str = error_str.decode("utf-8")
        raise RuntimeError(f"CNCL error: {error_str}")


_c_cnclGetVersion = cncl.cnclGetLibVersion
_c_cnclGetVersion.restype = ctypes.c_int
_c_cnclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]


def cnclGetVersion() -> str:
    major = ctypes.c_int()
    minor = ctypes.c_int()
    patch = ctypes.c_int()
    CNCL_CHECK(_c_cnclGetVersion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)))
    return "{}.{}.{}".format(str(major.value), str(minor.value), str(patch.value))


class CnclUniqueId(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128), ("hash", ctypes.c_uint64)]


_c_cnclGetUniqueId = cncl.cnclGetCliqueId
_c_cnclGetUniqueId.restype = ctypes.c_int
_c_cnclGetUniqueId.argtypes = [ctypes.POINTER(CnclUniqueId)]


def cnclGetUniqueId() -> CnclUniqueId:
    unique_id = CnclUniqueId()
    CNCL_CHECK(_c_cnclGetUniqueId(ctypes.byref(unique_id)))
    return unique_id


_c_cnclCommInitRank = cncl.cnclInitComms
_c_cnclCommInitRank.restype = ctypes.c_int
_c_cnclCommInitRank.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),    # comms
    ctypes.c_int,                       # num_comm
    ctypes.POINTER(ctypes.c_int),       # dev_list
    ctypes.POINTER(ctypes.c_int),       # rank_list
    ctypes.c_int,                       # nrank
    ctypes.POINTER(CnclUniqueId)        # clique_id
]

cnclDataType_t = ctypes.c_int


class cnclDataTypeEnum:
    cnclInvalid = 0x0
    cnclHalf = 0x10
    cnclFloat16 = 0x10
    cnclBfloat16 = 0x11
    cnclFloat = 0x12
    cnclFloat32 = 0x12
    cnclInt8 = 0x20
    cnclChar = 0x20
    cnclInt16 = 0x21
    cnclInt = 0x22
    cnclInt32 = 0x22
    cnclInt64 = 0x23
    cnclUint8 = 0x30
    cnclUint16 = 0x31
    cnclUint32 = 0x32
    cnclUint64 = 0x33
    cnclNumTypes = 51

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.cnclInt8
        if dtype == torch.uint8:
            return cls.cnclUint8
        if dtype == torch.int32:
            return cls.cnclInt32
        if dtype == torch.int64:
            return cls.cnclInt64
        if dtype == torch.float16:
            return cls.cnclFloat16
        if dtype == torch.float32:
            return cls.cnclFloat32
        # if dtype == torch.float64:
        #     return cls.cnclFloat64
        if dtype == torch.bfloat16:
            return cls.cnclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


cnclRedOp_t = ctypes.c_int


class cnclRedOpTypeEnum:
    cnclSum = 0
    cnclProd = 1
    cnclMax = 2
    cnclMin = 3
    # cnclAvg = 4
    cnclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.cnclSum
        if op == ReduceOp.PRODUCT:
            return cls.cnclProd
        if op == ReduceOp.MAX:
            return cls.cnclMax
        if op == ReduceOp.MIN:
            return cls.cnclMin
        # if op == ReduceOp.AVG:
        #     return cls.cnclAvg
        raise ValueError(f"Unsupported op: {op}")


_c_cnclAllReduce = cncl.cnclAllReduce
_c_cnclAllReduce.restype = ctypes.c_int
_c_cnclAllReduce.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cnclRedOp_t,
    cnclDataType_t, ctypes.c_void_p, ctypes.c_void_p
]


_c_cnclCommDestroy = cncl.cnclDestroyComms
_c_cnclCommDestroy.restype = ctypes.c_int
_c_cnclCommDestroy.argtypes = [ctypes.c_void_p, ctypes.c_int]


class CNCLCommunicator:

    def __init__(
        self,
        group: Optional[ProcessGroup] = None,
        device: Optional[Union[int, str, torch.device]] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CNCLCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        assert dist.is_initialized()
        group = get_cpu_world_group() if group is None else group
        assert dist.get_backend(group) != dist.Backend.CNCL, (
            "CNCLCommunicator should be attached to a non-CNCL group.")
        self.group = group
        # note: this rank is the rank in the group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.rank == 0:
            self.unique_id = cnclGetUniqueId()
        else:
            self.unique_id = CnclUniqueId()
        tensor = torch.ByteTensor(list(self.unique_id.data))
        ranks = dist.get_process_group_ranks(group)
        # arg `src` in `broadcast` is the global rank
        dist.broadcast(tensor, src=ranks[0], group=group)
        byte_list = tensor.tolist()
        for i, byte in enumerate(byte_list):
            self.unique_id.data[i] = byte
        self.comm = ctypes.c_void_p()
        if device is None:
            local_rank = get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # cncl communicator and stream will use this device
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        dev = ctypes.c_int(local_rank)
        rank = ctypes.c_int(self.rank)
        CNCL_CHECK(
            _c_cnclCommInitRank(ctypes.byref(self.comm), 1, ctypes.byref(dev), ctypes.byref(rank), self.world_size,
                                    ctypes.byref(self.unique_id)))
        self.stream = torch.cuda.Stream(device)

    def all_reduce(self,
                   tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None):
        # cncl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert tensor.device == self.device, (
            f"this cncl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self.stream
        CNCL_CHECK(
            _c_cnclAllReduce(ctypes.c_void_p(tensor.data_ptr()),
                             ctypes.c_void_p(tensor.data_ptr()),
                             tensor.numel(),
                             cnclDataTypeEnum.from_torch(tensor.dtype),
                             cnclRedOpTypeEnum.from_torch(op), self.comm,
                             ctypes.c_void_p(stream.mlu_stream)))
