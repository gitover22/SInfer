import datetime
import json
import logging
import os
import sys
from functools import partial
from logging import Logger
from logging.config import dictConfig
from os import path
from typing import Dict, Optional

import sinfer.envs as envs

SINFER_CONFIGURE_LOGGING = envs.SINFER_CONFIGURE_LOGGING # 是否配置日志记录系统
SINFER_LOGGING_CONFIG_PATH = envs.SINFER_LOGGING_CONFIG_PATH # 日志配置文件的路径

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"      # 时间戳格式

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "sinfer": {
            "class": "sinfer.logging.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "sinfer": {
            "class": "logging.StreamHandler",
            "formatter": "sinfer",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "sinfer": {
            "handlers": ["sinfer"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
}

# 初始化根日志器的配置
def _configure_sinfer_root_logger() -> None:
    logging_config: Optional[Dict] = None

    if not SINFER_CONFIGURE_LOGGING and SINFER_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "SINFER_CONFIGURE_LOGGING evaluated to false, but "
            "SINFER_LOGGING_CONFIG_PATH was given. SINFER_LOGGING_CONFIG_PATH "
            "implies SINFER_CONFIGURE_LOGGING. Please enable "
            "SINFER_CONFIGURE_LOGGING or unset SINFER_LOGGING_CONFIG_PATH.")

    if SINFER_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGING_CONFIG

    if SINFER_LOGGING_CONFIG_PATH:
        if not path.exists(SINFER_LOGGING_CONFIG_PATH):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                SINFER_LOGGING_CONFIG_PATH)
        # 从文件中加载自定义日志配置
        with open(SINFER_LOGGING_CONFIG_PATH, encoding="utf-8",
                  mode="r") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError("Invalid logging config. Expected Dict, got %s.",
                             type(custom_config).__name__)
        logging_config = custom_config

    if logging_config:
        dictConfig(logging_config) # 用一个dict配置logging


def init_logger(name: str) -> Logger:
    return logging.getLogger(name)

# 根日志器在模块导入时初始化。  
# 这是线程安全的，因为模块只会被导入一次，  
# 由Python GIL（全局解释器锁）保证。  
_configure_sinfer_root_logger()

logger = init_logger(__name__)

# 记录 root_dir 范围内所有函数的调用和返回
def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ['call', 'return']:
        # 提取文件名, 行号, 函数名
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir): # 如果文件路径不在 root_dir 范围内，跳过记录
            return
        try:
            last_frame = frame.f_back # 获取上一层调用栈信息
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # 如果是初始帧，设置为空值
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            # 打开日志文件，以追加方式写入函数调用或返回信息
            with open(log_path, 'a') as f:
                if event == 'call':
                    f.write(f"{datetime.datetime.now()} Call to"
                            f" {func_name} in {filename}:{lineno}"
                            f" from {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
                else:
                    f.write(f"{datetime.datetime.now()} Return from"
                            f" {func_name} in {filename}:{lineno}"
                            f" to {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
        except NameError:
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str,
                               root_dir: Optional[str] = None):
    """
    启用对 root_dir 下代码中每个函数调用的跟踪, 这对于调试程序挂起或崩溃非常有用
    log_file_path: 是日志文件的路径
    root_dir: 是需要跟踪的代码的根目录. 如果为 None, 则默认为 sinfer 的根目录 

    此调用是线程级的, 任何调用此函数的线程都会启用跟踪, 其他线程不会受到影响 
    """
    logger.warning(
        "SINFER_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only.")
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # by default, this is the sinfer root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))
