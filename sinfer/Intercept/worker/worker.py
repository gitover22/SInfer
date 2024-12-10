import os
import time
import torch
import sinfer
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.logger import init_logger
from sinfer.worker.worker import Worker, init_worker_distributed_environment
from sinfer.model_executor import set_random_seed

import sinfer.envs as envs

import functools
from collections import defaultdict

from sinfer.Intercept.model_executor.layers.feed_forward import FeedForward
from sinfer.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from sinfer.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear)

logger = init_logger(__name__)


def intercept_Worker_init_device(self) -> None:
    if self.device_config.device.type == "cuda":
        # torch.distributed.all_reduce 在同步点之前不会释放输入张量
        # 这会导致随着 all_reduce 调用次数的增加，内存使用量不断增长。
        # 通过设置该环境变量可以禁用这种行为
        # NCCL -> CNCL
        os.environ["TORCH_CNCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ.pop("CNCL_ASYNC_ERROR_HANDLING", None)  # 这个环境变量是由 Ray 设置的，它会导致图形构建时出现异常
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        intercept_check_if_gpu_supports_dtype(self.model_config.dtype)
        torch.cuda.empty_cache()
        self.init_gpu_memory = torch.cuda.mem_get_info()[0]
    else:
        raise RuntimeError(
            f"Not support device type: {self.device_config.device}")
    # 初始化分布式环境
    init_worker_distributed_environment(self.parallel_config, self.rank,
                                        self.distributed_init_method,
                                        self.local_rank)
    # 随机种子
    set_random_seed(self.model_config.seed)

def intercept_Worker_warm_up_model(self) -> None:
    # add num_gpu_blocks for mlu graph capture
    if not self.model_config.enforce_eager:
        self.model_runner.capture_model(self.gpu_cache, self.cache_config.num_gpu_blocks)
    
    set_random_seed(self.model_config.seed)


def intercept_Worker_get_latency(self):
    '''
    requires that torch.cuda.synchronize() be executed before this function 
    for getting an accurate reading
    '''
    start, end = self.model_runner.time_markers
    return start.elapsed_time(end)


def intercept_check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # for mlu
    logger.info(f"SInfer is using dtype: {torch_dtype}")
    
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 5:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")

def intercept_Worker_stat_tensor(name, tensor, act_range, key, dim):
    logger.debug(f"name:{name}, key:{key}, dim:{dim}, tensor.shape:{tensor.shape}")
    hidden_dim = tensor.shape[-1]
    # TODO
    # For torch.max has bug which generates nan/inf, so load the tensor to cpu to do torch.max.
    # And need to convert to mlu after the bug is fixed.
    # The pytorch jira: http://jira.cambricon.com/browse/PYTORCH-12199
    tensor = tensor.view(-1, hidden_dim).abs().cpu()
    comming_max = torch.max(tensor, dim=dim)[0].float()

    if act_range[name][key] is None:
        act_range[name][key] = comming_max
    else:
        act_range[name][key] = torch.max(act_range[name][key], comming_max)


def intercept_Worker_stat_input_hook(m, x, y, name, act_range, is_linear, is_save_input_id):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    logger.debug(f"name:{name}, x.shape:{x.shape}, y.shape:{y.shape}, m.weight.shape:{m.weight.shape}")
    if is_linear:
        intercept_Worker_stat_tensor(name, x, act_range, "x", 0)
        if act_range[name]["is_qkv"] and is_save_input_id and ".0." in name:
            x_cpu = x.clone().to("cpu")
            act_range[name]["input_id"].append(x_cpu)


def intercept_Worker_setup_smooth_hook(self, is_save_input_id: bool = False):
    model = self.model_runner.model
    self.act_range = defaultdict(lambda: {
        "x": None,
        "split": None,
        "is_linear": False,
        "is_qkv": False,
        "q_proj_size": 0,
        "num_kv_head_replicas": 1,
        "is_merge": False,
        "input_id": []
    })
    self.hooks = []
    linear_class_list = (ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
    other_class_list = (VocabParallelEmbedding, ParallelLMHead)
    class_list = linear_class_list + other_class_list
    row_class_list = (RowParallelLinear)

    for name, m in model.named_modules():
        if isinstance(m, FeedForward):
            m.use_bt_ffn = False

        if isinstance(m, class_list):
            is_linear = True if isinstance(m, linear_class_list) else False
            split_type = "row" if isinstance(m, row_class_list) else "col"
            self.act_range[name]["split"] = split_type
            self.act_range[name]["is_linear"] = is_linear
            if isinstance(m, QKVParallelLinear):
                self.act_range[name]["is_qkv"] = True
                self.act_range[name]["q_proj_size"] = m.num_heads * m.head_size
                self.act_range[name]["num_kv_head_replicas"] = m.num_kv_head_replicas
            self.act_range[name]["is_merge"] = isinstance(m, MergedColumnParallelLinear)

            logger.info(f"rank:{self.rank}, add hook to {name}, is_linear:{is_linear}, split_type:{split_type}")
            self.hooks.append(m.register_forward_hook(functools.partial(intercept_Worker_stat_input_hook,
                                                                        name=name, act_range=self.act_range,
                                                                        is_linear=is_linear,
                                                                        is_save_input_id=is_save_input_id)))


def intercept_Worker_remove_hooks(self):
    for h in self.hooks:
        h.remove()


def intercept_Worker_get_act_range(self):
    act_range = defaultdict(lambda: {
        "x": None,
        "split": None,
        "is_linear": False,
        "is_qkv": False,
        "q_proj_size": 0,
        "num_kv_head_replicas": 1,
        "is_merge": False,
        "input_id": []
    })
    for layer_name, layer_range in self.act_range.items():
        for tensor_key, tensor_value in layer_range.items():
            if isinstance(tensor_value, torch.Tensor):
                act_range[layer_name][tensor_key] = tensor_value.to("cpu")
            elif tensor_key == "input_id" and isinstance(tensor_value, list):
                input_id_len = len(tensor_value)
                for i in range(input_id_len):
                    if isinstance(tensor_value[i], torch.Tensor):
                        act_range[layer_name][tensor_key].append(tensor_value[i].to("cpu"))
                    else:
                        act_range[layer_name][tensor_key].append(tensor_value[i])
            else:
                act_range[layer_name][tensor_key] = tensor_value

    return act_range


def intercept_Worker_get_named_parameters(self):
    name_parameters = {}
    for name, param in self.model_runner.model.named_parameters():
        name_parameters[name] = param.to("cpu")

    return name_parameters


InterceptObject.add_intercept(Worker,
                                  Worker.init_device,
                                  intercept_Worker_init_device)
InterceptObject.add_intercept(Worker,
                                  Worker._warm_up_model,
                                  intercept_Worker_warm_up_model)
InterceptObject.add_intercept(Worker,
                                  "get_latency",
                                  intercept_Worker_get_latency)
InterceptObject.add_intercept(sinfer.worker.worker,
                                  sinfer.worker.worker._check_if_gpu_supports_dtype,
                                  intercept_check_if_gpu_supports_dtype)
InterceptObject.add_intercept(Worker,
                                  "stat_tensor",
                                  intercept_Worker_stat_tensor)
InterceptObject.add_intercept(Worker,
                                  "stat_input_hook",
                                  intercept_Worker_stat_input_hook)
InterceptObject.add_intercept(Worker,
                                  "setup_smooth_hook",
                                  intercept_Worker_setup_smooth_hook)
InterceptObject.add_intercept(Worker,
                                  "remove_hooks",
                                  intercept_Worker_remove_hooks)
InterceptObject.add_intercept(Worker,
                                  "get_act_range",
                                  intercept_Worker_get_act_range)
InterceptObject.add_intercept(Worker,
                                  "get_named_parameters",
                                  intercept_Worker_get_named_parameters)
