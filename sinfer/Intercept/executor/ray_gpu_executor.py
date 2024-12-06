import statistics
from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from sinfer.executor.ray_gpu_executor import RayGPUExecutor
from typing import Any, Dict
import os


def intercept_get_latency(self):
    '''
    调用此方法之前，必须执行 torch.cuda.synchronize()，以确保 CUDA 的操作完成（同步 GPU 和 CPU 操作），从而提供准确的延迟数据
    '''
    all_latencies = self._run_workers(method="get_latency")

    # 只有主驱动节点（driver worker）完成了 model.compute_logits() 过程，而其他 Ray 工作者（workers）在计算各自对应的 logit 后便停止运行
    return all_latencies[0]

def intercept_configure_ray_workers_use_nsight(self,
                                          ray_remote_kwargs) -> Dict[str, Any]:
    # 如果启用了 nsight 性能分析，我们需要为 ray workers 设置性能分析配置作为运行时环境
    runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})

    # 获取cnperf data 在每一个mlu的worker上
    runtime_env.update({
        "nsight": {
            "o": os.getcwd(),
            "force_overwrite": "false"
        }
    })
    return ray_remote_kwargs

InterceptObject.add_intercept(RayGPUExecutor,
                                  "get_latency",
                                  intercept_get_latency)
InterceptObject.add_intercept(RayGPUExecutor,
                                  "_configure_ray_workers_use_nsight",
                                  intercept_configure_ray_workers_use_nsight)