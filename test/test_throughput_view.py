import matplotlib.pyplot as plt
import numpy as np
# 该数据由benchmarks/benchmark_throughput.py跑出
# python benchmark_throughput.py --async-engine --model ~/Meta-Llama-3-8B-Instruct/ --tensor_parallel_size 2 --pipeline_parallel_size 2
data = {
    "TP": [26.62, 4259.81, 3407.84],
    "PP": [23.28, 3724.19, 2979.36],
    "TP+PP": [22.35, 3576.00, 2860.80]
}

categories = ["Requests/s", "Total Tokens/s", "Output Tokens/s"]
labels = list(data.keys())
values = np.array(list(data.values())).T

x = np.arange(len(labels))
width = 0.4

for i, category in enumerate(categories):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values[i], width, color=['blue', 'green', 'red'])
    
    ax.set_xlabel("Inference Methods")
    ax.set_ylabel("Values")
    ax.set_title(f"Comparison of {category}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    plt.savefig(f"{category.replace(' ', '_').replace('/', '_').lower()}_comparison.png")

    plt.show()