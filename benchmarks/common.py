import logging
from logging import Logger

def init_logger(name: str) -> Logger:
    """Initialize loggers for benchmarks module,
    and keep the configuration consistent with the sinfer module"""

    logger = logging.getLogger(name)

    sinfer_logger = logging.Logger.manager.loggerDict.get('sinfer', None)
    if sinfer_logger:
        logger.setLevel(sinfer_logger.level)
        logger.propagate = sinfer_logger.propagate
        logger.handlers = sinfer_logger.handlers

    return logger

