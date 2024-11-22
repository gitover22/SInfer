import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    SINFER_HOST_IP: str = ""
    SINFER_USE_MODELSCOPE: bool = False
    SINFER_INSTANCE_ID: Optional[str] = None
    SINFER_NCCL_SO_PATH: Optional[str] = None
    LD_LIBRARY_PATH: Optional[str] = None
    SINFER_USE_TRITON_FLASH_ATTN: bool = False
    LOCAL_RANK: int = 0
    MLU_VISIBLE_DEVICES: Optional[str] = None
    SINFER_ENGINE_ITERATION_TIMEOUT_S: int = 60
    SINFER_API_KEY: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    SINFER_CONFIG_ROOT: str = ""
    SINFER_USAGE_STATS_SERVER: str = "https://github.com/gitover22"
    SINFER_NO_USAGE_STATS: bool = False
    SINFER_DO_NOT_TRACK: bool = False
    SINFER_USAGE_SOURCE: str = ""
    SINFER_CONFIGURE_LOGGING: int = 1
    SINFER_LOGGING_CONFIG_PATH: Optional[str] = None
    SINFER_TRACE_FUNCTION: int = 0
    SINFER_ATTENTION_BACKEND: Optional[str] = None
    SINFER_CPU_KVCACHE_SPACE: int = 0
    SINFER_USE_RAY_COMPILED_DAG: bool = False
    SINFER_WORKER_MULTIPROC_METHOD: str = "spawn"
    SINFER_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: Optional[str] = None
    NVCC_THREADS: Optional[str] = None
    SINFER_BUILD_WITH_NEURON: bool = False
    SINFER_USE_PRECOMPILED: bool = False
    SINFER_INSTALL_PUNICA_KERNELS: bool = False
    CMAKE_BUILD_TYPE: Optional[str] = None
    VERBOSE: bool = False

environment_variables: Dict[str, Callable[[], Any]] = {
    # Target device of SInfer, supporting [cuda (by default), rocm, neuron, cpu]
    "SINFER_TARGET_DEVICE":
    lambda: os.getenv("SINFER_TARGET_DEVICE", "cuda"),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, sinfer will build with Neuron support
    "SINFER_BUILD_WITH_NEURON":
    lambda: bool(os.environ.get("SINFER_BUILD_WITH_NEURON", False)),

    # If set, sinfer will use precompiled binaries (*.so)
    "SINFER_USE_PRECOMPILED":
    lambda: bool(os.environ.get("SINFER_USE_PRECOMPILED")),

    # If set, sinfer will install Punica kernels
    "SINFER_INSTALL_PUNICA_KERNELS":
    lambda: bool(int(os.getenv("SINFER_INSTALL_PUNICA_KERNELS", "0"))),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, sinfer will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for SINFER configuration files
    # Note that this not only affects how sinfer finds its configuration files
    # during runtime, but also affects how sinfer installs its configuration
    # files during **installation**.
    "SINFER_CONFIG_ROOT":
    lambda: os.environ.get("SINFER_CONFIG_ROOT", None) or os.getenv(
        "XDG_CONFIG_HOME", None) or os.path.expanduser("~/.config"),

    # ================== Runtime Env Vars ==================

    # used in distributed environment to determine the master address
    'SINFER_HOST_IP':
    lambda: os.getenv('SINFER_HOST_IP', "") or os.getenv("HOST_IP", ""),

    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "SINFER_USE_MODELSCOPE":
    lambda: os.environ.get("SINFER_USE_MODELSCOPE", "False").lower() == "true",

    # Instance id represents an instance of the SINFER. All processes in the same
    # instance should have the same instance id.
    "SINFER_INSTANCE_ID":
    lambda: os.environ.get("SINFER_INSTANCE_ID", None),

    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),

    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "SINFER_NCCL_SO_PATH":
    lambda: os.environ.get("SINFER_NCCL_SO_PATH", None),

    # when `SINFER_NCCL_SO_PATH` is not set, sinfer will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # flag to control if sinfer should use triton flash attention
    "SINFER_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get("SINFER_USE_TRITON_FLASH_ATTN", "True").lower() in
             ("true", "1")),

    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # used to control the visible devices in the distributed setting
    "MLU_VISIBLE_DEVICES":
    lambda: os.environ.get("MLU_VISIBLE_DEVICES", None),

    # timeout for each iteration in the engine
    "SINFER_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("SINFER_ENGINE_ITERATION_TIMEOUT_S", "60")),

    # API key for SINFER API server
    "SINFER_API_KEY":
    lambda: os.environ.get("SINFER_API_KEY", None),

    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID":
    lambda: os.environ.get("S3_ACCESS_KEY", None),
    "S3_SECRET_ACCESS_KEY":
    lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL":
    lambda: os.environ.get("S3_ENDPOINT_URL", None),

    # Usage stats collection
    "SINFER_USAGE_STATS_SERVER":
    lambda: os.environ.get("SINFER_USAGE_STATS_SERVER", "https://github.com/gitover22"),
    "SINFER_NO_USAGE_STATS":
    lambda: os.environ.get("SINFER_NO_USAGE_STATS", "0") == "1",
    "SINFER_DO_NOT_TRACK":
    lambda: (os.environ.get("SINFER_DO_NOT_TRACK", None) or os.environ.get(
        "DO_NOT_TRACK", None) or "0") == "1",
    "SINFER_USAGE_SOURCE":
    lambda: os.environ.get("SINFER_USAGE_SOURCE", "production"),

    # Logging configuration
    # If set to 0, sinfer will not configure logging
    # If set to 1, sinfer will configure logging using the default configuration
    #    or the configuration file specified by SINFER_LOGGING_CONFIG_PATH
    "SINFER_CONFIGURE_LOGGING":
    lambda: int(os.getenv("SINFER_CONFIGURE_LOGGING", "1")),
    "SINFER_LOGGING_CONFIG_PATH":
    lambda: os.getenv("SINFER_LOGGING_CONFIG_PATH"),

    # Trace function calls
    # If set to 1, sinfer will trace function calls
    # Useful for debugging
    "SINFER_TRACE_FUNCTION":
    lambda: int(os.getenv("SINFER_TRACE_FUNCTION", "0")),

    # Backend for attention computation
    # Available options:
    # - "TORCH_SDPA": use torch.nn.MultiheadAttention
    # - "FLASH_ATTN": use FlashAttention
    # - "XFORMERS": use XFormers
    # - "ROCM_FLASH": use ROCmFlashAttention
    "SINFER_ATTENTION_BACKEND":
    lambda: os.getenv("SINFER_ATTENTION_BACKEND", None),

    # CPU key-value cache space
    # default is 4GB
    "SINFER_CPU_KVCACHE_SPACE":
    lambda: int(os.getenv("SINFER_CPU_KVCACHE_SPACE", "0")),

    # If the env var is set, it uses the Ray's compiled DAG API
    # which optimizes the control plane overhead.
    # Run SInfer with SINFER_USE_RAY_COMPILED_DAG=1 to enable it.
    "SINFER_USE_RAY_COMPILED_DAG":
    lambda: bool(os.getenv("SINFER_USE_RAY_COMPILED_DAG", 0)),

    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "SINFER_WORKER_MULTIPROC_METHOD":
    lambda: os.getenv("SINFER_WORKER_MULTIPROC_METHOD", "spawn"),
}

def __getattr__(name):
    # 惰性求值
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
