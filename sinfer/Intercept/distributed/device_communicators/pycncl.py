import ctypes
import platform
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sinfer.distributed.parallel_state import get_cpu_world_group, get_local_rank
from sinfer.logger import init_logger
from sinfer.Intercept.mlu_hijack_utils import find_cncl_library

logger = init_logger(__name__)

so_file = find_cncl_library()   # /usr/local/neuware/lib64/libcncl.so

try:
    cncl = ctypes.CDLL(so_file) # 使用另一个进程动态加载cncl库，实现对底层 C 接口的调用
except Exception as e:
    logger.error(
        "Failed to load CNCL library from %s ."
        "It is expected if you are not running on CAMBRICON MLUs.",
        so_file,
        platform.platform())
    raise e

cnclResult_t = ctypes.c_int


_c_cnclGetErrorString = cncl.cnclGetErrorStr # 函数绑定，python与C函数进行绑定
_c_cnclGetErrorString.restype = ctypes.c_char_p # 返回值类型
_c_cnclGetErrorString.argtypes = [cnclResult_t]

# 使用 CNCL_CHECK包装对CNCL函数调用返回值的检查 
def CNCL_CHECK(result: cnclResult_t) -> None:
    if result != 0:
        error_str = _c_cnclGetErrorString(result)
        error_str = error_str.decode("utf-8")
        raise RuntimeError(f"CNCL error: {error_str}")


_c_cnclGetVersion = cncl.cnclGetLibVersion
_c_cnclGetVersion.restype = ctypes.c_int
_c_cnclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

# 获取CNCL版本号
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

# 获取Clique的ID
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

# 将 CNCL 的 cnclAllReduce 函数从 C 接口映射到 Python
_c_cnclAllReduce = cncl.cnclAllReduce
_c_cnclAllReduce.restype = ctypes.c_int   # 返回值类型
# 定义函数参数类型： 源张量的内存地址，目标张量的内存地址，张量的元素数量，归约操作的类型，张量的数据类型，通信器的指针，流的指针
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
                             cnclRedOpTypeEnum.from_torch(op), 
                             self.comm,
                             ctypes.c_void_p(stream.mlu_stream)))
