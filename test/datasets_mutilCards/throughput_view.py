import matplotlib.pyplot as plt
import numpy as np
# 该数据由benchmarks/benchmark_throughput.py跑出
# python benchmark_throughput.py --async-engine --model ~/Meta-Llama-3-8B-Instruct/ --tensor_parallel_size 2 --pipeline_parallel_size 2
# concurrent: 16
# 1*MLU370 : Throughput: 1.56 requests/s, 225.02 total tokens/s, 200.02 output tokens/s
# 2*MLU370 : Throughput: 2.34 requests/s, 336.51 total tokens/s, 299.12 output tokens/s
# 4*MLU370 : Throughput: 2.69 requests/s, 387.87 total tokens/s, 344.77 output tokens/s

# concurrent: 32
# 1*MLU370 : Throughput: 2.94 requests/s, 423.06 total tokens/s, 376.05 output tokens/s
# 2*MLU370 : Throughput: 4.10 requests/s, 589.70 total tokens/s, 524.18 output tokens/s
# 4*MLU370 : Throughput: 4.75 requests/s, 684.16 total tokens/s, 608.14 output tokens/s

# concurrent: 64
# 1*MLU370 : Throughput: 5.13 requests/s, 738.09 total tokens/s, 656.08 output tokens/s
# 2*MLU370 : Throughput: 7.05 requests/s, 1015.90 total tokens/s, 903.03 output tokens/s
# 4*MLU370 : Throughput: 7.87 requests/s, 1133.12 total tokens/s, 1007.21 output tokens/s

# concurrent: 128
# 1*MLU370 : Throughput: 5.04 requests/s, 726.31 total tokens/s, 645.61 output tokens/s
# 2*MLU370 : Throughput: 11.27 requests/s, 1623.30 total tokens/s, 1442.93 output tokens/s
# 4*MLU370 : Throughput: 12.01 requests/s, 1729.50 total tokens/s, 1537.34 output tokens/s

# concurrent: 256
# 1*MLU370 : Throughput: 4.99 requests/s, 719.16 total tokens/s, 639.25 output tokens/s
# 2*MLU370 : Throughput: 15.05 requests/s, 2167.60 total tokens/s, 1926.75 output tokens/s
# 4*MLU370 : Throughput: 16.84 requests/s, 2424.77 total tokens/s, 2155.35 output tokens/s

# concurrent: 512
# 1*MLU370 : Throughput: 4.30 requests/s, 619.38 total tokens/s, 550.56 output tokens/s
# 2*MLU370 : Throughput: 15.31 requests/s, 2204.41 total tokens/s, 1959.48 output tokens/s
# 4*MLU370 : Throughput: 20.11 requests/s, 2895.41 total tokens/s, 2573.70 output tokens/s


data_16 = {
    "1*MLU370": [1.56, 225.02, 200.02],
    "2*MLU370": [2.34, 336.51, 299.12],
    "4*MLU370": [2.69, 387.87, 344.77],
}

data_32 = {
    "1*MLU370": [2.94, 423.06, 376.05],
    "2*MLU370": [4.10, 589.70, 524.18],
    "4*MLU370": [4.75, 684.16, 608.14],
}

data_64 = {
    "1*MLU370": [5.13, 738.09, 656.08],
    "2*MLU370": [7.05, 1015.90, 903.03],
    "4*MLU370": [7.87, 1133.12, 1007.21],
}

data_128 = {
    "1*MLU370": [5.04, 726.31, 645.61],
    "2*MLU370": [11.27, 1623.30, 1442.93],
    "4*MLU370": [12.01, 1729.50, 1537.34],
}

data_256 = {
    "1*MLU370": [4.99, 719.16, 639.25],
    "2*MLU370": [15.05, 2167.60, 1926.75],
    "4*MLU370": [16.84, 2424.77, 2155.35],
}

data_512 = {
    "1*MLU370": [4.30, 619.38, 550.56],
    "2*MLU370": [15.31, 2204.41, 1959.48],
    "4*MLU370": [20.11, 2895.41, 2573.70],
}

categories = ["Requests/s", "Total Tokens/s", "Output Tokens/s"]
labels = list(data_512.keys())
values = np.array(list(data_512.values())).T  # 转置以便按类别分组

x = np.arange(len(labels))  # x轴位置
width = 0.6  # 柱状图宽度

# 遍历每个类别，绘制对应的柱状图
for i, category in enumerate(categories):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, values[i], width, color=['blue', 'green', 'red'])
    
    # 在每个柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x坐标：柱子中心
            height,  # y坐标：柱子顶部
            f'{height:.2f}',  # 显示的数值，保留两位小数
            ha='center',  # 水平居中
            va='bottom'  # 垂直底部对齐
        )
    
    # 设置图表标题和标签
    ax.set_xlabel("MLU370 Configuration (Concurrent Requests = 512)")
    ax.set_ylabel(f"{category}")
    ax.set_title(f"Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # 保存图表
    plt.savefig(f"{category.replace(' ', '_').replace('/', '_').lower()}_comparison_512.png")
    
    # 显示图表
    plt.show()