import os
import time
import torch
import sinfer
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
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


def sinfer__worker__worker__Worker__init_device(self) -> None:
    if self.device_config.device.type == "cuda":
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573

        # change NCCL -> CNCL
        os.environ["TORCH_CNCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("CNCL_ASYNC_ERROR_HANDLING", None)
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        sinfer__worker__worker___check_if_gpu_supports_dtype(self.model_config.dtype)
        torch.cuda.empty_cache()
        self.init_gpu_memory = torch.cuda.mem_get_info()[0]
    else:
        raise RuntimeError(
            f"Not support device type: {self.device_config.device}")
    # Initialize the distributed environment.
    init_worker_distributed_environment(self.parallel_config, self.rank,
                                        self.distributed_init_method,
                                        self.local_rank)
    # Set random seed.
    set_random_seed(self.model_config.seed)


def sinfer__worker__worker__Worker___warm_up_model(self) -> None:

    # add num_gpu_blocks for mlu graph capture
    if not self.model_config.enforce_eager:
        self.model_runner.capture_model(self.gpu_cache, self.cache_config.num_gpu_blocks)
    
    set_random_seed(self.model_config.seed)


def sinfer__worker__worker__Worker__get_latency(self):
    '''
    requires that torch.cuda.synchronize() be executed before this function 
    for getting an accurate reading
    '''
    start, end = self.model_runner.time_markers
    return start.elapsed_time(end)


def sinfer__worker__worker___check_if_gpu_supports_dtype(torch_dtype: torch.dtype):

    # modify capability check for mlu
    logger.info(f"SInfer is using dtype: {torch_dtype}")
    # Check if the GPU supports the dtype.
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

def sinfer__worker__worker__Worker__stat_tensor(name, tensor, act_range, key, dim):
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


def sinfer__worker__worker__Worker__stat_input_hook(m, x, y, name, act_range, is_linear, is_save_input_id):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    logger.debug(f"name:{name}, x.shape:{x.shape}, y.shape:{y.shape}, m.weight.shape:{m.weight.shape}")
    if is_linear:
        sinfer__worker__worker__Worker__stat_tensor(name, x, act_range, "x", 0)
        if act_range[name]["is_qkv"] and is_save_input_id and ".0." in name:
            x_cpu = x.clone().to("cpu")
            act_range[name]["input_id"].append(x_cpu)


def sinfer__worker__worker__Worker__setup_smooth_hook(self, is_save_input_id: bool = False):
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
            self.hooks.append(m.register_forward_hook(functools.partial(sinfer__worker__worker__Worker__stat_input_hook,
                                                                        name=name, act_range=self.act_range,
                                                                        is_linear=is_linear,
                                                                        is_save_input_id=is_save_input_id)))


def sinfer__worker__worker__Worker__remove_hooks(self):
    for h in self.hooks:
        h.remove()


def sinfer__worker__worker__Worker__get_act_range(self):
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


def sinfer__worker__worker__Worker__get_named_parameters(self):
    name_parameters = {}
    for name, param in self.model_runner.model.named_parameters():
        name_parameters[name] = param.to("cpu")

    return name_parameters


MluHijackObject.add_hijack_object(Worker,
                                  Worker.init_device,
                                  sinfer__worker__worker__Worker__init_device)
MluHijackObject.add_hijack_object(Worker,
                                  Worker._warm_up_model,
                                  sinfer__worker__worker__Worker___warm_up_model)
MluHijackObject.add_hijack_object(Worker,
                                  "get_latency",
                                  sinfer__worker__worker__Worker__get_latency)
MluHijackObject.add_hijack_object(sinfer.worker.worker,
                                  sinfer.worker.worker._check_if_gpu_supports_dtype,
                                  sinfer__worker__worker___check_if_gpu_supports_dtype)

MluHijackObject.add_hijack_object(Worker,
                                  "stat_tensor",
                                  sinfer__worker__worker__Worker__stat_tensor)
MluHijackObject.add_hijack_object(Worker,
                                  "stat_input_hook",
                                  sinfer__worker__worker__Worker__stat_input_hook)
MluHijackObject.add_hijack_object(Worker,
                                  "setup_smooth_hook",
                                  sinfer__worker__worker__Worker__setup_smooth_hook)
MluHijackObject.add_hijack_object(Worker,
                                  "remove_hooks",
                                  sinfer__worker__worker__Worker__remove_hooks)
MluHijackObject.add_hijack_object(Worker,
                                  "get_act_range",
                                  sinfer__worker__worker__Worker__get_act_range)
MluHijackObject.add_hijack_object(Worker,
                                  "get_named_parameters",
                                  sinfer__worker__worker__Worker__get_named_parameters)
