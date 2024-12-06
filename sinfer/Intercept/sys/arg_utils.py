import argparse
import torch
from sinfer.config import RuntimeConfig
from sinfer.sys.arg_utils import RuntimeArgs, AsyncRuntimeArgs
from sinfer.Intercept.mlu_definitions import BlockSizeInfo, USE_PAGED
from sinfer.Intercept.mlu_hijack_utils import InterceptObject, set_batch_sizes_to_capture
from sinfer.logger import init_logger

logger = init_logger(__name__)


create_engine_config_org = RuntimeArgs.create_runtime_config
add_cli_args_org = RuntimeArgs.add_cli_args
from_cli_args_org = RuntimeArgs.from_cli_args
from_cli_args_org = AsyncRuntimeArgs.from_cli_args
post_init__org = RuntimeArgs.__post_init__

def intercept_post_init_(self,) -> None:
    # In MLU3XX device, when the tensor_parallel_size > 1, the enforce_eager is forced to set False
    unsupport_device = "3" in torch.cuda.get_device_name(0)
    if unsupport_device and self.tensor_parallel_size > 1 and self.enforce_eager == False:
        self.enforce_eager = True
        logger.warning("The current device only support enforce_eager = True, when the tensor_parallel_size > 1. "
                       "The param enforce_eager is forced to set True")
    post_init__org(self)
    
def intercept_create_runtime_config(self, ) -> RuntimeConfig:
    # disable custom_all_reduce, re-set block_size to support paged and unpaged mode
    # MLU not support custom all reduce
    self.disable_custom_all_reduce = True
    BlockSizeInfo.set_block_size(self.block_size)
    if not USE_PAGED and self.enable_chunked_prefill:
        raise ValueError("Don't support chunked_prefill in UnPaged mode.")

    engine_config = create_engine_config_org(self)
    engine_config.cache_config.block_size = BlockSizeInfo.BLOCK_SIZE
    
    return engine_config


@staticmethod
def intercept_add_cli_args(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """通用CLI命令行接口"""
    parser = add_cli_args_org(parser)
    # remove block_size choices
    for action in parser._actions:
        if action.dest == "block_size":
            action.choices = None
            action.default = -1  # 设为-1

    parser.add_argument('--max-batch_size-to-capture',
                        type=int,
                        default=None,
                        help='The max batch size to capture MLUGraph.')
    return parser


@classmethod
def intercept_from_cli_args(cls, args: argparse.Namespace) -> 'RuntimeArgs':
    max_batch_size_to_capture = getattr(args, "max_batch_size_to_capture", None)
    if max_batch_size_to_capture:
        set_batch_sizes_to_capture(max_batch_size_to_capture)
    if cls == AsyncRuntimeArgs:
        return from_cli_args_org(args)
    else:
        return from_cli_args_org(args)


InterceptObject.add_intercept(RuntimeArgs,
                                  RuntimeArgs.__post_init__,
                                  intercept_post_init_)
InterceptObject.add_intercept(RuntimeArgs,
                                  RuntimeArgs.create_runtime_config,
                                  intercept_create_runtime_config)
InterceptObject.add_intercept(RuntimeArgs,
                                  RuntimeArgs.add_cli_args,
                                  intercept_add_cli_args)
InterceptObject.add_intercept(RuntimeArgs,
                                  RuntimeArgs.from_cli_args,
                                  intercept_from_cli_args)
