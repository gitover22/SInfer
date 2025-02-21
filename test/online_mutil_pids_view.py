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

# 使用示例
log_file1 = "./log/TP-50.log"  # 请确保文件路径正确
log_file2 = "./log/TP-100.log"  # 请确保文件路径正确
log_file3 = "./log/TP-200.log"  # 请确保文件路径正确
res1 = calculate_avg_e2e(log_file1)
res2 = calculate_avg_e2e(log_file2)
res3 = calculate_avg_e2e(log_file3)

# 可视化结果
labels = ['TP-50', 'TP-100', 'TP-200']
values = [res1, res2, res3]

plt.bar(labels, values, color=['blue', 'green', 'red'])
plt.xlabel('Parallel mode')
plt.ylabel('Average e2e Latency (ms)')
plt.title('MLUX4 * 2 TP')
plt.savefig(f'e2e_latency.png')
plt.show()
