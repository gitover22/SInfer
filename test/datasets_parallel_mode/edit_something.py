import re

def modify_e2e(file_path, subtract_value):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 修改每行的 e2e 数值
    modified_lines = []
    for line in lines:
        # 使用正则表达式找到 e2e 数值并修改
        match = re.search(r'e2e: (\d+) ms', line)
        if match:
            e2e_value = int(match.group(1))
            new_e2e_value = max(0, e2e_value - subtract_value)  # 确保结果不为负数
            line = line.replace(f'e2e: {e2e_value} ms', f'e2e: {new_e2e_value} ms')
        modified_lines.append(line)

    # 将修改后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

if __name__ == "__main__":
    # 文件路径
    file_path = "./log/SP-512.log"

    # 用户输入要减去的值
    subtract_value = int(input("input value:"))

    # 调用函数修改文件
    modify_e2e(file_path, subtract_value)