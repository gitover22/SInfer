import matplotlib.pyplot as plt
import numpy as np
# 该数据由benchmarks/benchmark_throughput.py跑出
# ==========================================================
# python benchmark_throughput.py --async-engine --model ~/Meta-Llama-3-8B-Instruct/ --tensor_parallel_size 2 --pipeline_parallel_size 2
# ==========================================================



# 4* 370X4   concurrent: 64
# TP : Throughput: 9.08 requests/s, 1306.90 total tokens/s, 1161.69 output tokens/s
# PP : Throughput: 5.04 requests/s, 726.40 total tokens/s, 645.69 output tokens/s
# HP : Throughput: 7.44 requests/s, 1071.99 total tokens/s, 952.88 output tokens/s
# SP : Throughput: 7.87 requests/s, 1133.12 total tokens/s, 1007.21 output tokens/s

# 4* 370X4   concurrent: 128
# TP : Throughput: 14.55 requests/s, 2095.52 total tokens/s, 1862.68 output tokens/s
# PP : Throughput: 8.86 requests/s, 1275.28 total tokens/s, 1133.59 output tokens/s
# HP : Throughput: 11.85 requests/s, 1706.99 total tokens/s, 1517.33 output tokens/s
# SP : Throughput: 12.01 requests/s, 1729.50 total tokens/s, 1537.34 output tokens/s

# 4* 370X4   concurrent: 256
# TP : Throughput: 14.29 requests/s, 2057.90 total tokens/s, 1829.24 output tokens/s
# PP : Throughput: 8.87 requests/s, 1277.60 total tokens/s, 1135.64 output tokens/s
# HP : Throughput: 12.41 requests/s, 1786.74 total tokens/s, 1588.21 output tokens/s
# SP : Throughput: 16.84 requests/s, 2424.77 total tokens/s, 2155.35 output tokens/s

# 4* 370X4   concurrent: 512
# TP : Throughput: 17.95 requests/s, 2585.45 total tokens/s, 2298.18 output tokens/s 
# PP : Throughput: 8.46 requests/s, 1217.97 total tokens/s, 1082.64 output tokens/s
# HP : Throughput: 15.05 requests/s, 2167.31 total tokens/s, 1926.49 output tokens/s
# SP : Throughput: 20.11 requests/s, 2895.41 total tokens/s, 2573.70 output tokens/s


# 4* 370X4   concurrent: 64
data64 = {
    "TP-64": [9.08, 1306.90, 1161.69],
    "PP-64": [5.04, 726.40, 645.69],
    "HP-64": [7.44, 1071.99, 952.88],
    "SP-64": [7.87, 1133.12, 1007.21],
}

# 4* 370X4   concurrent: 128
data128 = {
    "TP-128": [14.55, 2095.52, 1862.68],
    "PP-128": [8.86, 1275.28, 1133.59],
    "HP-128": [11.85, 1706.99, 1517.33],
    "SP-128": [12.01, 1729.50, 1537.34],
}

# 4* 370X4   concurrent: 256
data256 = {
    "TP-256": [14.29, 2057.90, 1829.24],
    "PP-256": [8.87, 1277.60, 1135.64],
    "HP-256": [12.41, 1786.74, 1588.21],
    "SP-256": [16.84, 2424.77, 2155.35],
}

# 4* 370X4   concurrent: 512
data512 = {
    "TP-512": [17.95, 2585.45, 2298.18],
    "PP-512": [8.46, 1217.97, 1082.64],
    "HP-512": [15.05, 2167.31, 1926.49],
    "SP-512": [20.11, 2895.41, 2573.70],
}

categories = ["Requests/s", "Total Tokens/s", "Output Tokens/s"]
labels = list(data256.keys())
values = np.array(list(data256.values())).T  # 转置以便按类别分组

x = np.arange(len(labels))  # x轴位置
width = 0.6  # 柱状图宽度

# 遍历每个类别，绘制对应的柱状图
for i, category in enumerate(categories):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, values[i], width, color=['blue', 'green', 'red', 'yellow'])
    
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
    ax.set_xlabel("Parallel mode(max concurrent requests)")
    ax.set_ylabel(f"{category}")
    ax.set_title(f"Throughput - {category}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # 保存图表
    plt.savefig(f"{category.replace(' ', '_').replace('/', '_').lower()}_comparison_256.png")
    
    # 显示图表
    plt.show()