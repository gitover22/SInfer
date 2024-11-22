import statistics
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.executor.ray_gpu_executor import RayGPUExecutor
from typing import Any, Dict
import os


def sinfer__executor__ray_gpu_executor__RayGPUExecutor__get_latency(self):
    '''
    requires that torch.cuda.synchronize() be executed before this function 
    for getting an accurate reading
    '''
    all_latencies = self._run_workers(method="get_latency")

    # only the driver worker finishes the model.compute_logits() process whereas
    # the ray workers rest after computing each respective logit.
    return all_latencies[0]

def sinfer__executor__ray_gpu_executor__RayGPUExecutor___configure_ray_workers_use_nsight(self,
                                          ray_remote_kwargs) -> Dict[str, Any]:
    # If nsight profiling is enabled, we need to set the profiling
    # configuration for the ray workers as runtime env.
    runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
    # get cnperf data on each worker of mlu
    runtime_env.update({
        "nsight": {
            "o": os.getcwd(),
            "force_overwrite": "false"
        }
    })
    return ray_remote_kwargs

MluHijackObject.add_hijack_object(RayGPUExecutor,
                                  "get_latency",
                                  sinfer__executor__ray_gpu_executor__RayGPUExecutor__get_latency)
MluHijackObject.add_hijack_object(RayGPUExecutor,
                                  "_configure_ray_workers_use_nsight",
                                  sinfer__executor__ray_gpu_executor__RayGPUExecutor___configure_ray_workers_use_nsight)