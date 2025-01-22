import argparse
import json
import random
import time
from typing import List, Optional
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化日志记录
def init_logger(name):
    import logging
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = init_logger(__name__)

# 分布式初始化
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 模拟输入数据
def generate_fake_data(batch_size: int, seq_length: int) -> List[str]:
    fake_data = [" ".join([f"token{random.randint(0, 100)}" for _ in range(seq_length)]) for _ in range(batch_size)]
    return fake_data

# 推理函数
def run_inference(model, tokenizer, inputs: List[str], max_length: int):
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**tokenized_inputs, max_length=max_length)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# 主函数
def main(rank, world_size, model_name, batch_size, seq_length, max_length):
    setup(rank, world_size)

    logger.info(f"Rank {rank}: Initializing model and tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(rank)
    model = DDP(model, device_ids=[rank])

    fake_data = generate_fake_data(batch_size, seq_length)
    logger.info(f"Rank {rank}: Running inference.")

    results = run_inference(model, tokenizer, fake_data, max_length)

    logger.info(f"Rank {rank}: Inference completed. Here are some results:")
    for i, result in enumerate(results[:5]):  # 打印前5个结果
        logger.info(f"Rank {rank} - Result {i + 1}: {result}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Inference Script")
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3-8B-Instruct", help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=32, help="Sequence length of input")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of generated sequences")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of MLUs to use")

    args = parser.parse_args()

    world_size = args.world_size
    torch.multiprocessing.spawn(
        main,
        args=(world_size, args.model_name, args.batch_size, args.seq_length, args.max_length),
        nprocs=world_size,
        join=True
    )
