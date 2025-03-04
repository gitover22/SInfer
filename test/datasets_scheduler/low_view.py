import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
sinfer_data = pd.read_csv('sinfer_tas_low.csv')
vllm_data = pd.read_csv('vllm_fcfs_low.csv')

# 设置绘图风格
plt.style.use('default')

# 横坐标：max_concurrent_requests
x = sinfer_data['max_concurrent_requests']
bar_width = 0.35  # 柱形宽度
x_indices = np.arange(len(x))  # 用于定位柱形的位置

# 绘制 TTFT 图
plt.figure(figsize=(10, 6))
plt.bar(
    x_indices - bar_width / 2,
    sinfer_data['TTFT'],
    width=bar_width,
    label='SInfer TAS',
    color='red',
    alpha=0.8
)
plt.bar(
    x_indices + bar_width / 2,
    vllm_data['TTFT'],
    width=bar_width,
    label='vLLM FCFS',
    color='blue',
    alpha=0.8
)
plt.title('TTFT', fontsize=16)
plt.ylabel('TTFT (s)', fontsize=14)
plt.xlabel('Max Concurrent Requests', fontsize=14)
plt.xticks(x_indices, x)  # 设置横坐标标签
# plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('low_view_TTFT_bar.png', dpi=300)  # 保存 TTFT 图表
plt.show()

# 绘制 TPOT 图
plt.figure(figsize=(10, 6))
plt.bar(
    x_indices - bar_width / 2,
    sinfer_data['TPOT'],
    width=bar_width,
    label='SInfer TAS',
    color='red',
    alpha=0.8
)
plt.bar(
    x_indices + bar_width / 2,
    vllm_data['TPOT'],
    width=bar_width,
    label='vLLM FCFS',
    color='blue',
    alpha=0.8
)
plt.title('TPOT', fontsize=16)
plt.ylabel('TPOT (s)', fontsize=14)
plt.xlabel('Max Concurrent Requests', fontsize=14)
plt.xticks(x_indices, x)  # 设置横坐标标签
# plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('low_view_TPOT_bar.png', dpi=300)  # 保存 TPOT 图表
plt.show()