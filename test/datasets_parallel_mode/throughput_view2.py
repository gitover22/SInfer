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
# TP : Throughput: 14.55 requests/s, 2075.52 total tokens/s, 1842.68 output tokens/s
# PP : Throughput: 8.86 requests/s, 1275.28 total tokens/s, 1133.59 output tokens/s
# HP : Throughput: 11.85 requests/s, 1706.99 total tokens/s, 1517.33 output tokens/s
# SP : Throughput: 13.71 requests/s, 1969.50 total tokens/s, 1737.34 output tokens/s

# 4* 370X4   concurrent: 256
# TP : Throughput: 15.29 requests/s, 2157.90 total tokens/s, 1929.24 output tokens/s
# PP : Throughput: 8.91 requests/s, 1279.60 total tokens/s, 1135.64 output tokens/s
# HP : Throughput: 12.41 requests/s, 1786.74 total tokens/s, 1588.21 output tokens/s
# SP : Throughput: 16.84 requests/s, 2424.77 total tokens/s, 2155.35 output tokens/s

# 4* 370X4   concurrent: 512
# TP : Throughput: 17.95 requests/s, 2495.45 total tokens/s, 2298.18 output tokens/s 
# PP : Throughput: 9.26 requests/s, 1317.97 total tokens/s, 1182.64 output tokens/s
# HP : Throughput: 15.05 requests/s, 2167.31 total tokens/s, 1926.49 output tokens/s
# SP : Throughput: 20.11 requests/s, 2805.41 total tokens/s, 2573.70 output tokens/s


# 4* 370X4   concurrent: 64
data64 = {
    "TP-64": [9.08, 1306.90, 1161.69],
    "PP-64": [5.04, 726.40, 645.69],
    "HP-64": [7.44, 1071.99, 952.88],
    "SP-64": [7.87, 1133.12, 1007.21],
}

# 4* 370X4   concurrent: 128
data128 = {
    "TP-128": [14.55, 2075.52, 1842.68],
    "PP-128": [8.86, 1275.28, 1133.59],
    "HP-128": [11.85, 1706.99, 1517.33],
    "SP-128": [13.71, 1969.50, 1737.34],
}

# 4* 370X4   concurrent: 256
data256 = {
    "TP-256": [15.29, 2157.90, 1929.24], 
    "PP-256": [8.91, 1279.60, 1135.64],  
    "HP-256": [12.41, 1786.74, 1588.21],
    "SP-256": [16.84, 2424.77, 2155.35],
}

# 4* 370X4   concurrent: 512
data512 = {
    "TP-512": [17.95, 2585.45, 2298.18],
    "PP-512": [9.26, 1317.97, 1182.64],
    "HP-512": [15.05, 2167.31, 1926.49],
    "SP-512": [20.11, 2795.41, 2573.70], 
}
# 合并所有数据
all_data = [data64, data128, data256, data512]

# 计算每个类别的最大值
categories = ["Requests/s", "Total Tokens/s", "Output Tokens/s"]
max_values = {category: 0 for category in categories}

for data in all_data:
    for values in data.values():
        for i, category in enumerate(categories):
            if values[i] > max_values[category]:
                max_values[category] = values[i]

# 设置统一的 y 轴范围（最大值的 1.1 倍）
y_limits = {category: max_values[category] * 1.05 for category in categories}

# 绘制图表
def plot_data(data, concurrent):
    # labels = list(data.keys())
    labels =["TP","PP","HP","SP"]
    values = np.array(list(data.values())).T  # 转置以便按类别分组
    x = np.arange(len(labels))  # x轴位置
    width = 0.6  # 柱状图宽度

    # 遍历每个类别，绘制对应的柱状图
    for i, category in enumerate(categories):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(x, values[i], width, color=['blue', 'green', 'red', 'yellow'])
        
        # # 在每个柱子上方添加数值标签
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(
        #         bar.get_x() + bar.get_width() / 2,  # x坐标：柱子中心
        #         height,  # y坐标：柱子顶部
        #         f'{height:.2f}',  # 显示的数值，保留两位小数
        #         ha='center',  # 水平居中
        #         va='bottom'  # 垂直底部对齐
        #     )
        
        # 设置图表标题和标签
        ax.set_xlabel("Parallel Mode")
        ax.set_ylabel(f"{category}")
        ax.set_title(f"Throughput - {category} (Concurrent: {concurrent})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, y_limits[category])  # 设置统一的 y 轴范围
        
        # 保存图表
        plt.savefig(f"{category.replace(' ', '_').replace('/', '_').lower()}_comparison_{concurrent}_V2_NoValue.png")
        
        # 显示图表
        plt.show()

# 绘制每个并发数的图表
plot_data(data64, 64)
plot_data(data128, 128)
plot_data(data256, 256)
plot_data(data512, 512)