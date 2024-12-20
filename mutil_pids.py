import os
import time
from datetime import datetime
from multiprocessing import Process, Lock
import logging
from tzlocal import get_localzone
def get_LocalTime():

    # 获取本地时区
    local_timezone = get_localzone()

    # 获取本地时间
    local_time = datetime.now(local_timezone)

    # 格式化本地时间
    time = local_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return time

# 配置日志模块
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)  # 创建日志目录

current_time = get_LocalTime()
log_file = os.path.join(log_dir, f"pid_logs_{current_time}.log")  # 根据时间动态生成日志文件名

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
        model="./test/Meta-Llama-3-8B-Instruct",
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


# 记录日志函数
def log_process_info(process_id, pid, start_time, end_time, runtime):
    log_message = (
        f"user id: {process_id} | pid: {pid} | start: {start_time} | "
        f"end: {end_time} | live time: {runtime} ms"
    )
    logger.info(log_message)  # 使用自定义日志记录器记录日志
    print(log_message)


# 模拟运行的函数
def run_process(process_id):
    start_time = get_LocalTime()
    start_ts = int(time.time() * 1000)  # 毫秒时间戳

    print(f"用户id: {process_id} | 创建时间: {start_time}")

    start()

    # 获取进程销毁时间（精确到毫秒）
    end_time = get_LocalTime()
    end_ts = int(time.time() * 1000)  # 毫秒时间戳

    # 计算运行时长
    runtime = end_ts - start_ts

    # 获取当前进程ID
    pid = os.getpid()

    # 记录日志
    log_process_info(process_id, pid, start_time, end_time, runtime)


def main(process_count):
    processes = []

    # 创建指定数量的进程
    for i in range(1, process_count + 1):
        p = Process(target=run_process, args=(i,))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("所有进程完成")


if __name__ == "__main__":
    pid_count = int(input("num pid: "))
    main(pid_count)
