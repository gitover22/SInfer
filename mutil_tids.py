import threading
import time
import os 
from datetime import datetime
import logging

# 配置日志模块
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)  # 创建日志目录

# 获取当前时间，格式化为字符串（精确到秒）
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"tid_logs_{current_time}.log")  # 根据时间动态生成日志文件名

# 创建自定义日志记录器
logger = logging.getLogger("custom_logger")
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(file_handler)
logger.propagate = False  # 禁止传播到全局日志器，避免记录其他模块的日志


def start():
    from openai import OpenAI

    # 初始化 OpenAI 客户端
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    # 定义用户输入
    user_message = {"role": "user", "content": "Write a python program. Implements integer summation,You just need to output the program."}

    # 请求模型生成结果
    completion = client.chat.completions.create(
        model="/home/zouguoqiang/SInfer/models/Meta-Llama-3-8B-Instruct",
        messages=[user_message],
        max_tokens=1000  # 确保生成内容不超过上下文窗口限制
    )

    # 打印用户输入和模型生成内容
    print("="*50)
    print("User Prompt:", user_message["content"])
    print("="*50)
    print("Model Response:", completion.choices[0].message.content)
    print("="*50)
    print("\n\n")

def run_prompt(user_id):
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    native_id = threading.current_thread().native_id  # 获取真实线程 ID
    print(f"user id: {user_id} | tid: {native_id} | start: {start_time}")
    start_ts = int(time.time() * 1000)  # 毫秒时间戳
    start()
    end_ts = int(time.time() * 1000)  # 毫秒时间戳
    # 计算运行时长
    runtime = end_ts - start_ts
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_message = (
        f"user id: {user_id} | tid: {native_id} | start: {start_time} | "
        f"end: {end_time} | live time: {runtime} ms"
    )
    logger.info(log_message)  # 记录线程结束日志
    print(log_message)

if __name__ == "__main__":
    thread_count = int(input("num threads: "))
    threads = []

    for i in range(thread_count):
        t = threading.Thread(target=run_prompt, args=(i + 1,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    print("all threads end.")
