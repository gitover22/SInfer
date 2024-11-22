from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.executor.gpu_executor import GPUExecutor


def sinfer__executor__gpu_executor__GPUExecutor__get_latency(self):
    '''
    requires that torch.cuda.synchronize() be executed before this function 
    for getting an accurate reading
    '''
    latency = self.driver_worker.get_latency()
    return latency


MluHijackObject.add_hijack_object(GPUExecutor,
                                  "get_latency",
                                  sinfer__executor__gpu_executor__GPUExecutor__get_latency)