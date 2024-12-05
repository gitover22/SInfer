import argparse
import torch
from sinfer.config import EngineConfig
from sinfer.sys.arg_utils import EngineArgs, AsyncEngineArgs
from sinfer.mlu_hijack.mlu_definitions import BlockSizeInfo, USE_PAGED
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject, set_batch_sizes_to_capture
from sinfer.logger import init_logger

logger = init_logger(__name__)


sinfer__engine__arg_utils__EngineArgs__create_engine_config_org = EngineArgs.create_engine_config
sinfer__engine__arg_utils__EngineArgs__add_cli_args_org = EngineArgs.add_cli_args
sinfer__engine__arg_utils__EngineArgs__from_cli_args_org = EngineArgs.from_cli_args
sinfer__engine__arg_utils__AsyncEngineArgs__from_cli_args_org = AsyncEngineArgs.from_cli_args
sinfer_engine__arg_utils__EngineArgs____post_init__org = EngineArgs.__post_init__

def sinfer_engine__arg_utils__EngineArgs____post_init__(self,) -> None:
    # In MLU3XX device, when the tensor_parallel_size > 1, the enforce_eager is forced to set False
    unsupport_device = "3" in torch.cuda.get_device_name(0)
    if unsupport_device and self.tensor_parallel_size > 1 and self.enforce_eager == False:
        self.enforce_eager = True
        logger.warning("The current device only support enforce_eager = True, when the tensor_parallel_size > 1. "
                       "The param enforce_eager is forced to set True")
    sinfer_engine__arg_utils__EngineArgs____post_init__org(self)
    
def sinfer__engine__arg_utils__EngineArgs__create_engine_config(self, ) -> EngineConfig:
    # disable custom_all_reduce, re-set block_size to support paged and unpaged mode
    # MLU not support custom all reduce
    self.disable_custom_all_reduce = True
    BlockSizeInfo.set_block_size(self.block_size)
    if not USE_PAGED and self.enable_chunked_prefill:
        raise ValueError("Don't support chunked_prefill in UnPaged mode.")

    engine_config = sinfer__engine__arg_utils__EngineArgs__create_engine_config_org(self)
    engine_config.cache_config.block_size = BlockSizeInfo.BLOCK_SIZE
    
    return engine_config


@staticmethod
def sinfer__engine__arg_utils__EngineArgs__add_cli_args(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """通用CLI命令行接口"""
    parser = sinfer__engine__arg_utils__EngineArgs__add_cli_args_org(parser)
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
def sinfer__engine__arg_utils__EngineArgs__from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
    max_batch_size_to_capture = getattr(args, "max_batch_size_to_capture", None)
    if max_batch_size_to_capture:
        set_batch_sizes_to_capture(max_batch_size_to_capture)
    if cls == AsyncEngineArgs:
        return sinfer__engine__arg_utils__AsyncEngineArgs__from_cli_args_org(args)
    else:
        return sinfer__engine__arg_utils__EngineArgs__from_cli_args_org(args)


MluHijackObject.add_hijack_object(EngineArgs,
                                  EngineArgs.__post_init__,
                                  sinfer_engine__arg_utils__EngineArgs____post_init__)
MluHijackObject.add_hijack_object(EngineArgs,
                                  EngineArgs.create_engine_config,
                                  sinfer__engine__arg_utils__EngineArgs__create_engine_config)
MluHijackObject.add_hijack_object(EngineArgs,
                                  EngineArgs.add_cli_args,
                                  sinfer__engine__arg_utils__EngineArgs__add_cli_args)
MluHijackObject.add_hijack_object(EngineArgs,
                                  EngineArgs.from_cli_args,
                                  sinfer__engine__arg_utils__EngineArgs__from_cli_args)
