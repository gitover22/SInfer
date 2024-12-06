from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from sinfer.executor.gpu_executor import GPUExecutor


def intercept_get_latency(self):
    '''
    requires that torch.cuda.synchronize() be executed before this function 
    for getting an accurate reading
    '''
    latency = self.driver_worker.get_latency()
    return latency


InterceptObject.add_intercept(GPUExecutor,
                                  "get_latency",
                                  intercept_get_latency)