import torch

def test():
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    print(f"Free GPU memory: {free_gpu_memory / 1024 / 1024:.2f} MB")
if __name__ == "__main__":
    test()