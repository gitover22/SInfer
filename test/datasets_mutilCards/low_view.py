import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
cards1_data = pd.read_csv('1cards_low.csv')
cards4_data = pd.read_csv('4cards_low.csv')

# 设置绘图风格
plt.style.use('default')

# 横坐标：max_concurrent_requests
x = cards1_data['max_concurrent_requests']
bar_width = 0.35  # 柱形宽度
x_indices = np.arange(len(x))  # 用于定位柱形的位置

# 绘制 TTFT 图
plt.figure(figsize=(10, 6))
plt.bar(
    x_indices - bar_width / 2,
    cards1_data['TTFT'],
    width=bar_width,
    label='1*MLU370',
    color='red',
    alpha=0.8
)
plt.bar(
    x_indices + bar_width / 2,
    cards4_data['TTFT'],
    width=bar_width,
    label='4*MLU370',
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
    cards1_data['TPOT'],
    width=bar_width,
    label='1*MLU370',
    color='red',
    alpha=0.8
)
plt.bar(
    x_indices + bar_width / 2,
    cards4_data['TPOT'],
    width=bar_width,
    label='4*MLU370',
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