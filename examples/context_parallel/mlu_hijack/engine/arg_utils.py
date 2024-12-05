import argparse
import torch
from sinfer.config import EngineConfig, ParallelConfig
from sinfer.sys.arg_utils import EngineArgs, AsyncEngineArgs
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject, set_batch_sizes_to_capture
from sinfer.mlu_hijack.sys.arg_utils import sinfer__engine__arg_utils__EngineArgs__create_engine_config
from sinfer.mlu_hijack.sys.arg_utils import sinfer__engine__arg_utils__EngineArgs__add_cli_args
from sinfer.mlu_hijack.sys.arg_utils import sinfer__engine__arg_utils__EngineArgs__from_cli_args

sinfer__engine__arg_utils__EngineArgs__from_cli_args_org = EngineArgs.from_cli_args
sinfer__engine__arg_utils__AsyncEngineArgs__from_cli_args_org = AsyncEngineArgs.from_cli_args

def sinfer__engine__arg_utils__EngineArgs__create_engine_config_wraper(self, ) -> EngineConfig:
    '''
    ==========================
    Modify by Context Parallel
    ==========================
    @brief: set parallel_config context_parallel_size
    '''
    if self.context_parallel_size > 1 and ("3" in torch.cuda.get_device_name(0)):
        raise ValueError('Context parallel does not support MLU370.')
    
    setattr(ParallelConfig, 'context_parallel_size', self.context_parallel_size)
    engine_config = sinfer__engine__arg_utils__EngineArgs__create_engine_config(self)
    '''
    =======================
    End of Context Parallel
    =======================
    '''
    return engine_config


@staticmethod
def sinfer__engine__arg_utils__EngineArgs__add_cli_args_wraper(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Shared CLI arguments for SInfer engine."""
    parser = sinfer__engine__arg_utils__EngineArgs__add_cli_args(parser)
    '''
    ==========================
    Modify by Context Parallel
    ==========================
    @brief: add context-parallel-size choices
    '''
    parser.add_argument('--context-parallel-size',
                        '-cp',
                        type=int,
                        default=EngineArgs.context_parallel_size,
                        help='number of context parallel replicas')
    '''
    =======================
    End of Context Parallel
    =======================
    '''
    return parser


@classmethod
def sinfer__engine__arg_utils__EngineArgs__from_cli_args_wraper(cls, args: argparse.Namespace) -> 'EngineArgs':
    engine_args = sinfer__engine__arg_utils__EngineArgs__from_cli_args(cls, args)
    setattr(engine_args, 'context_parallel_size', getattr(args, "context_parallel_size"))
    return engine_args

MluHijackObject.add_hijack_object(EngineArgs,
                                  EngineArgs.create_engine_config,
                                  sinfer__engine__arg_utils__EngineArgs__create_engine_config_wraper)
MluHijackObject.add_hijack_object(EngineArgs,
                                  EngineArgs.add_cli_args,
                                  sinfer__engine__arg_utils__EngineArgs__add_cli_args_wraper)
MluHijackObject.add_hijack_object(EngineArgs,
                                  EngineArgs.from_cli_args,
                                  sinfer__engine__arg_utils__EngineArgs__from_cli_args_wraper)