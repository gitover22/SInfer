"""
这个例子展示了如何使用Ray进行离线批量推理(分布在一个多节点的集群上)
Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""
from typing import Dict
import numpy as np
import ray

from sinfer import LLM, SamplingParams

# Create a class to do batch inference.
class LLMPredictor:
    def __init__(self):
        self.llm = LLM(model="/home/zouguoqiang/SInfer/test/Meta-Llama-3-8B-Instruct",
                       dtype="float16",
                       tensor_parallel_size =4,
                       gpu_memory_utilization=0.9
                       )
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        outputs = self.llm.generate(batch["text"], sampling_params)
        prompt = []
        generated_text = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 从S3读取文本 RayData支持从云存储中读取多个文件（如 JSONL、Parquet、CSV、二进制格式）
ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")

# 对所有输入数据进行批量推理
ds = ds.map_batches(
    LLMPredictor,
    # 将并发数设置为 LLM 实例的数量
    concurrency=2,
    # num_gpus：指定每个 LLM 实例所需的 GPU 数量
    # num_gpus=4,  # 不设置 num_gpus，因为 tensor_parallel_size 已经启用
    batch_size=32,
)

# 查看前10个结果（本地测试和调试） 在生产中应将完整结果输出
outputs = ds.take(limit=10)
for output in outputs:
    prompt = output["prompt"]
    generated_text = output["generated_text"]
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# 将推理输出数据写为 Parquet 文件到 S3
# 多个文件将写入目标位置，每个任务将单独写入一个或多个文件
# ds.write_parquet("s3://<your-output-bucket>")
