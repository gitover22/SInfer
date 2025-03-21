import re
import matplotlib.pyplot as plt

def calculate_avg_e2e(log_file):
    total_e2e = 0
    user_count = 0
    
    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(r'user id: \d+ \| pid: \d+ \| e2e: (\d+) ms', line)
            if match:
                total_e2e += int(match.group(1))
                user_count += 1
    
    if user_count == 0:
        print("No valid entries found in the log file.")
        return None
    
    avg_e2e = total_e2e / user_count
    print(log_file)
    print(f"Average e2e latency: {avg_e2e:.2f} ms")
    return avg_e2e


log_file1 = "./log/TP-512.log" 
log_file2 = "./log/PP-512.log" 
log_file3 = "./log/HP-512.log" 
log_file4 = "./log/SP-512.log" 
res1 = calculate_avg_e2e(log_file1)
res2 = calculate_avg_e2e(log_file2)
res3 = calculate_avg_e2e(log_file3)
res4 = calculate_avg_e2e(log_file4)

# 可视化
labels = ['TP', 'PP', 'HP','SP']
values = [res1, res2, res3, res4]

# 创建柱状图
bars = plt.bar(labels, values, color=['blue', 'green', 'red', 'yellow'])

# 在每个柱子上方添加数值标签
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2., height,
#              f'{height:.2f}',
#              ha='center', va='bottom')

plt.xlabel('Parallel Mode')
plt.ylabel('Latency (ms)')
plt.title('Latency (Concurrent: 512)')
plt.savefig('e2e_latency_V2_NoValue_512.png')
plt.show()