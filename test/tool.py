import logging
from logging import Logger
import csv
def read_prompts_from_csv(path, num):
    prompts = []
    with open(path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i >= num:
                break
            prompts.append(row['prompt'])
    return prompts

def init_logger(name: str) -> Logger:
    # 保持sinfer模块的配置一致
    logger = logging.getLogger(name)
    sinfer_logger = logging.Logger.manager.loggerDict.get('sinfer', None)
    if sinfer_logger:
        logger.setLevel(sinfer_logger.level)
        logger.propagate = sinfer_logger.propagate
        logger.handlers = sinfer_logger.handlers

    return logger
