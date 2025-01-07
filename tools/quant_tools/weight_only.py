import argparse
import torch
from torch import Tensor
import numpy as np
import logging

from sinfer import LLM

from utils_internal import convert_to_merged, exclude_expert_in_merged_act_range, cleanup
from dump_smooth import save_weights, save_generate_weights

logger = logging.getLogger(__name__)


def merge_adjacent_low_4bit(tensor: Tensor):
    """
    将一个包含int8类型数据的张量，按相邻两个元素的低4位合并成新的int8数据，
    并输出一个新的张量。

    参数:
    - tensor: 类型为torch.int8的张量，长度应为偶数。

    返回:
    - 新张量，其中每个元素是相邻原元素低4位的合并结果。

    示例:
    a = torch.tensor([5, 7, 12, 3], dtype=torch.int8)  # 示例张量，每对元素将被合并
    merged_tensor = merge_adjacent_low_nibbles(a)
    print(f"合并后的张量: {merged_tensor} (二进制: {merged_tensor.tolist()})")
    """

    # 确保输入张量类型为int8且长度为偶数
    assert tensor.dtype == torch.int8, "输入张量必须为int8类型"
    assert tensor.shape[-1] % 2 == 0, "输入张量最后一维长度需为偶数"

    even = np.bitwise_and(tensor[..., 0::2], 0x0F, dtype=np.int8)
    odd = np.bitwise_and(tensor[..., 1::2], 0x0F, dtype=np.int8)
    merged_tensor = np.bitwise_or(np.left_shift(odd, 4), even)

    # 结果是已经合并的新张量
    return merged_tensor


def cal_weightonly_weight(weight, weight_bits, qmin, qmax, has_qzeros, eps: float = 1e-8):
    '''
    return quantized_weight, scales, qzeros
    args:
        weight: need to be quantized
        weight_bits: quantized bitwidth
        qmin: minimum value in quantized range
        qmax: maximum value in quantized range
        has_qzeros: whether to generate qzeros weight
        eps: limit zero float value to avoid floatpoint error
    '''
    assert weight.numel() != 0, "weight should not be empty tensor"
    assert weight.dim() == 2 or weight.dim() == 3, "Invalid dim. The dim of weight should be 2 or 3"
    assert weight.dtype in [torch.float32, torch.float16, torch.bfloat16
                            ], "Invalid datatype. Weight must be torch.float32 or torch.float16 or torch.bfloat16"

    weight_scale = weight.float().abs().clamp(min=eps).max(dim=-1).values / qmax
    unpacked_weight = (torch.round((weight / weight_scale[..., None]).float())).clip(min=qmin, max=qmax).to(torch.int8)
    scale_quant_orig_c = weight_scale.squeeze()

    if weight_bits == 4:
        quantized_weight = merge_adjacent_low_4bit(unpacked_weight)
    else:
        quantized_weight = unpacked_weight

    if has_qzeros:
        qzeros = torch.zeros_like(scale_quant_orig_c, dtype=torch.int32)
    else:
        qzeros = None

    return quantized_weight, scale_quant_orig_c, qzeros


def generate_weightonly_weight(act_range, name_parameters, args):
    '''
    generate hugging face weight to quanizated weightonly weight
    args:
        act_range: non parallem act_range
        name_parameters: non parallel hugging face named parameters
        args: arguments from main
    '''
    weightonly_weight = {}
    has_qzeros = args.has_qzeros
    weight_bits = 8 if args.weight_only_precision == 'int8' else 4
    qmin = float(-2**(weight_bits - 1))
    qmax = float(2**(weight_bits - 1) - 1)

    for name, param in name_parameters.items():
        if name.endswith("bias"):
            weightonly_weight[name] = param
            continue
        name_parts = name.split(".")
        layer_name = ".".join(name_parts[:-1])
        if layer_name in act_range:
            qweight, scales, qzeros = cal_weightonly_weight(param, weight_bits, qmin, qmax, has_qzeros)
            scales = scales.to(args.torch_scales_smooth_dtype)
            weightonly_weight[f'{layer_name}.qweight'] = qweight
            weightonly_weight[f'{layer_name}.scales'] = scales
            if has_qzeros:
                weightonly_weight[f'{layer_name}.qzeros'] = qzeros
        else:
            weightonly_weight[name] = param

    return weightonly_weight


def generate_weights_of_weight_only(llm: LLM, args: argparse.Namespace):
    '''
    generate weightonly weights
    args:
        llm: LLM instance
        args: argument from main
    '''
    tp_size = args.tp_size

    if tp_size > 1:
        llm.llm_runtime._interfacer._run_workers("setup_smooth_hook")
    else:
        llm.llm_runtime._interfacer.driver_worker.setup_smooth_hook()

    if tp_size > 1:
        llm.llm_runtime._interfacer._run_workers("remove_hooks")
        act_range = llm.llm_runtime._interfacer._run_workers("get_act_range")
        named_parameters = llm.llm_runtime._interfacer._run_workers("get_named_parameters")
    else:
        llm.llm_runtime._interfacer.driver_worker.remove_hooks()
        act_range = llm.llm_runtime._interfacer.driver_worker.get_act_range()
        named_parameters = llm.llm_runtime._interfacer.driver_worker.get_named_parameters()

    del llm
    cleanup()

    logger.info("get act_range and named_parameters from llm finished")

    merged_act_range, merged_named_parameters, _ = convert_to_merged(act_range, named_parameters, tp_size, args)
    save_weights(merged_named_parameters, args)

    del act_range
    del named_parameters
    cleanup()

    logger.info("get merged_act_range and merged_named_parameters finished")

    if args.enable_quantize_expert is False:
        merged_act_range = exclude_expert_in_merged_act_range(merged_act_range)

    weightonly_weight = generate_weightonly_weight(merged_act_range, merged_named_parameters, args)
    save_generate_weights(weightonly_weight, args)

    del merged_act_range
    del merged_named_parameters
    cleanup()

    logger.info("get weightonly_weight finished")

    return weightonly_weight
