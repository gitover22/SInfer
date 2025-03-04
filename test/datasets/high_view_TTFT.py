import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
vllm_data = pd.read_csv('/home/huafeng/SInfer/test/datasets/vllm_fcfs_high.csv')
sinfer_data = pd.read_csv('/home/huafeng/SInfer/test/datasets/sinfer_tas_high.csv')

# 设置绘图风格
plt.style.use('default')

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制 vllm_fcfs 的折线图
plt.plot(
    vllm_data['max_concurrent_requests'],
    # vllm_data['TTFT'],
    vllm_data['TTFT'],
    label='vLLM FCFS',
    marker='o',
    linestyle='-',
    linewidth=2,
    color='blue'
)

# 绘制 sinfer_tas 的折线图
plt.plot(
    sinfer_data['max_concurrent_requests'],
    # sinfer_data['TTFT'],
    sinfer_data['TTFT'],
    label='SInfer TAS',
    marker='s',
    linestyle='--',
    linewidth=2,
    color='red'
)

# 设置标题和标签
# plt.title('TTFT', fontsize=16)
plt.title('TTFT', fontsize=16)
plt.xlabel('Max Concurrent Requests', fontsize=14)
# plt.ylabel('TTFT (s)', fontsize=14)
plt.ylabel('TTFT (s)', fontsize=14)

# 设置网格
# plt.grid(True, linestyle='--', alpha=0.6)

# 设置图例
plt.legend(fontsize=12)

# 设置横坐标为对数刻度（因为 max_concurrent_requests 跨度较大）
# plt.xscale('log')

# 显示图形
plt.tight_layout()
plt.savefig('high_view_TTFT.png')
plt.show()