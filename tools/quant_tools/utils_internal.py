from collections import defaultdict, OrderedDict
import torch
from pathlib import Path
from typing import Optional
import re
import os
import shutil
import logging
import json
from transformers import AutoTokenizer, T5Tokenizer
import gc
from datetime import datetime

from model_special import (smooth_model_config, get_layer_weight_bias_name, get_qkv_distribution,
                           modify_layer_weight_bias_name)

logger = logging.getLogger(__name__)


_str_to_torch_dtype_dict = dict(
    bfloat16=torch.bfloat16,
    float16=torch.float16,
    float32=torch.float32,
    int64=torch.int64,
    int32=torch.int32,
    int8=torch.int8,
    bool=torch.bool,
    fp8=torch.float8_e4m3fn,
)


def str_dtype_to_torch(dtype):
    '''
    convert torch dytpe to str dtype
    '''
    ret = _str_to_torch_dtype_dict.get(dtype)
    dtype = ret if ret is not None else torch.float16
    return dtype


_torch_dtype_to_str_dict = {
    torch.bfloat16:"bfloat16",
    torch.float16:"float16",
    torch.float32:"float32",
    torch.int64:"int64",
    torch.int32:"int32",
    torch.int8:"int8",
    torch.bool:"bool",
    torch.float8_e4m3fn:"fp8",
}


def torch_dtype_to_str(dtype):
    '''
    convert str dytpe to torch dtype
    '''
    ret = _torch_dtype_to_str_dict.get(dtype)
    dtype = ret if ret is not None else "float16"
    return dtype


def extract_model_path(name_or_path):
    '''
    extract model_version, model_family from named_or_path from config.json
    '''
    patterns = [
        r"/(.*)(-[0-9]+[mMbB]{1})(-*.*)",
        r"/(.*-[0-9]+)(-*.*)",
        r"(.*)(-[0-9]+[mMbB]{1})(-*.*)",
        r"(.*-[0-9]+)(-*.*)",
        r"([^-]+)(-*.*)",
    ]
    model_version = None
    for pattern in patterns:
        match = re.search(pattern, name_or_path)
        if match:
            model_version = match.group(1)
            break

    if model_version is None:
        model_version = name_or_path

    model_version = model_version.lower()
    match = re.search(r"([a-zA-z]+)(.*)", model_version)
    if match:
        model_family = match.group(1)
    else:
        model_family = model_version

    return model_version, model_family


def read_model_name(model_dir: str, model_version: Optional[str] = None, model_type: Optional[str] = None):
    '''
    get model_arch, model_version, model_family, model_type form config.json, passed model_version, model_type
    args:
        model_dir: model directory
        model_version: passed from main, default None
        model_type: pass from main, default None
    '''
    with open(Path(model_dir) / "config.json", 'r') as f:
        config = json.load(f)

    model_arch = config.get('architectures', None)
    name_or_path = config.get('_name_or_path', None)
    if model_type is None:
        model_type = config.get('model_type', None)
    if model_type:
        model_type = model_type.lower()
    model_family = None

    if model_version is None and name_or_path:
        model_version, model_family = extract_model_path(name_or_path)

    if model_version is None:
        model_version = model_type

    if model_version:
        model_version = model_version.lower()

    if model_version and model_family is None:
        match = re.search(r"([a-zA-z]+)(.*)", model_version)
        if match:
            model_family = match.group(1)
        else:
            model_family = model_version

    assert model_arch, "read model architectures failed"
    assert model_version, "read model version failed, please set args.version manually"
    assert model_family, "read model family failed, please set args.version manually"

    return model_arch, model_version, model_family, model_type


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    '''
    load tokenizer of model
    args:
        tokenizer_dir: tokenizer directory
        vocab_file: vocabulary file, default None
        model_name: model name
        model_version: model version
        tokenizer_type: Tokenizer type to be loaded.
    '''
    if vocab_file is None:
        use_fast = True
        if tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='right',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    elif model_name == 'GemmaForCausalLM':
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(vocab_file=vocab_file, padding_side='left', truncation_side='left', legacy=False)
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(vocab_file=vocab_file, padding_side='left', truncation_side='left', legacy=False)

    if model_name == 'QWenForCausalLM':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config['chat_format']
        assert chat_format in ('raw','chatml'), f"unknown chat format: {chat_format}"
        pad_id = gen_config['pad_token_id']
        end_id = gen_config['eos_token_id']
    elif model_name in ('ChatGLMForCausalLM', 'glm'):
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    try:
        tokenizer.pad_token = tokenizer.eos_token
    except AttributeError:
        pass

    return tokenizer, pad_id, end_id


def merge_qkv_weight(named_parameters, weight_name, tp_size, q_proj_size, num_kv_head_replicas):
    '''
    merge tensor parallel qkv weight to none parallel q_weight, k_weight, v_weight.
    merge_qkv weight and bias has the same logic
    args:
        named_parameters: parallel named parameters
        weight_name: qkv layer weight name
        tp_size: tensor parallel size
        q_proj_size: query projection size
        num_kv_head_replicas: number kv head replicas
    '''
    qkv_proj_size = named_parameters[0][weight_name].shape[0]
    kv_proj_size = (qkv_proj_size - q_proj_size) // 2
    splite_size = [q_proj_size, kv_proj_size, kv_proj_size]

    q_weight_list = []
    k_weight_list = []
    v_weight_list = []

    for rank in range(0, tp_size):
        weight = named_parameters[rank][weight_name]
        split_weight = torch.split(weight, splite_size, dim=0)
        q_weight_list.append(split_weight[0])
        if rank % num_kv_head_replicas == 0:
            k_weight_list.append(split_weight[1])
            v_weight_list.append(split_weight[2])

    q_weight = torch.cat(q_weight_list, dim=0)
    k_weight = torch.cat(k_weight_list, dim=0)
    v_weight = torch.cat(v_weight_list, dim=0)

    return q_weight, k_weight, v_weight


def merge_merged_weight(named_parameters, weight_name, tp_size, dim=0):
    '''
    merge merged linear layer weight to gate_weight and up_weight.
    merge merged weight and bias has the same logic.
    args:
        named_parameters: parallel named parameters
        weight_name: qkv layer weight name
        tp_size: tensor parallel size
    '''
    up_weight_list = []
    gate_weight_list = []

    for rank in range(0, tp_size):
        weight = named_parameters[rank][weight_name]
        chunk_weights = torch.chunk(weight, 2, dim=dim)
        up_weight_list.append(chunk_weights[0])
        gate_weight_list.append(chunk_weights[1])

    gate_weight = torch.cat(up_weight_list, dim=dim)
    up_weight = torch.cat(gate_weight_list, dim=dim)

    return gate_weight, up_weight


def convert_packed_qkv(q_weight, k_weight, v_weight, dim, args):
    '''
    convert packad qkv weight or bias
    args:
        q_weight: q weight or bias
        k_weight: k weight or bias
        v_weight: v_weight or bias
        dim: convert dim
        args: argument
    '''
    packed_qkv = torch.cat([q_weight, k_weight, v_weight], dim=dim)
    is_n3sh, head_num, kv_head_num = get_qkv_distribution(args.model_type, args.model_version, args.hf_config)
    if is_n3sh is True:
        packed_qkv_shape = packed_qkv.shape
        num_query_heads_per_kv_head = head_num // kv_head_num
        q_shape = q_weight.shape
        k_shape = k_weight.shape
        v_shape = v_weight.shape
        q = q_weight.view(q_shape[:dim] + (kv_head_num, num_query_heads_per_kv_head, -1) + q_shape[dim + 1:])
        k = k_weight.view(k_shape[:dim] + (kv_head_num, 1, -1) + k_shape[dim + 1:])
        v = v_weight.view(v_shape[:dim] + (kv_head_num, 1, -1) + v_shape[dim + 1:])
        tensor_n3sh = torch.cat([q, k, v], dim=dim+1)
        packed_qkv = tensor_n3sh.reshape(packed_qkv_shape)

    return packed_qkv


def convert_to_merged_qkv_weight(layer_name, weight_name, bias_name, named_parameters, merged_named_parameters,
                                 layer_range, merged_act_range, tp_size, args):
    '''
    convert parallel qkv named parameters to non parallel qkv named parameters
    args:
        layer_name: layer name
        weight_name: weight name
        bias_name: bias name
        named_parameters: parallel hugging face named parameters
        merged_named_parameters: non parallel hugging face named parameters
        layer_range: parallel layer range info
        merged_act_range: non parallel act range
        tp_size: tensor parallel size
        args: argument
    '''
    layer_name_parts = layer_name.split(".")
    self_attn_layer_name = ".".join(layer_name_parts[:-1])
    qkv_name = layer_name_parts[-1]
    q_weight, k_weight, v_weight = merge_qkv_weight(named_parameters, weight_name, tp_size, layer_range["q_proj_size"],
                                                    layer_range["num_kv_head_replicas"])
    qkv_list = smooth_model_config[args.model_type]["qkv_list"]
    qkv_list_len = len(qkv_list)
    if qkv_list_len == 3:
        q_layer_name = f"{self_attn_layer_name}.{qkv_list[0]}"
        k_layer_name = f"{self_attn_layer_name}.{qkv_list[1]}"
        v_layer_name = f"{self_attn_layer_name}.{qkv_list[2]}"
    elif qkv_list_len == 1:
        qkv_layer_name = f"{self_attn_layer_name}.{qkv_list[0]}"

    if qkv_list_len == 3:
        merged_act_range[q_layer_name]["x"] = layer_range["x"]
        merged_act_range[k_layer_name]["x"] = layer_range["x"]
        merged_act_range[v_layer_name]["x"] = layer_range["x"]
        merged_act_range[q_layer_name]["is_qkv"] = True
        merged_act_range[k_layer_name]["is_qkv"] = True
        merged_act_range[v_layer_name]["is_qkv"] = True

        merged_named_parameters[f"{q_layer_name}.weight"] = q_weight
        merged_named_parameters[f"{k_layer_name}.weight"] = k_weight
        merged_named_parameters[f"{v_layer_name}.weight"] = v_weight
    elif qkv_list_len == 1:
        merged_act_range[qkv_layer_name]["x"] = layer_range["x"]
        qkv_weight = convert_packed_qkv(q_weight, k_weight, v_weight, 0, args)
        merged_named_parameters[f"{qkv_layer_name}.weight"] = qkv_weight

    if bias_name in named_parameters[0]:
        q_bias, k_bias, v_bias = merge_qkv_weight(named_parameters, bias_name, tp_size, layer_range["q_proj_size"],
                                                  layer_range["num_kv_head_replicas"])
        if qkv_list_len == 3:
            merged_named_parameters[f"{q_layer_name}.bias"] = q_bias
            merged_named_parameters[f"{k_layer_name}.bias"] = k_bias
            merged_named_parameters[f"{v_layer_name}.bias"] = v_bias
        elif qkv_list_len == 1:
            qkv_bias = convert_packed_qkv(q_bias, k_bias, v_bias, 0, args)
            merged_named_parameters[f"{qkv_layer_name}.bias"] = qkv_bias

    return qkv_name


def convert_to_merged_merged_weight(layer_name, weight_name, bias_name, named_parameters, merged_named_parameters,
                                 layer_range, merged_act_range, tp_size, model_type):
    '''
    convert parallel merged named parameters to non parallel merged named parameters
    args:
        layer_name: layer name
        weight_name: weight name
        bias_name: bias name
        named_parameters: parallel hugging face named parameters
        merged_named_parameters: non parallel hugging face named parameters
        layer_range: parallel layer range info
        merged_act_range: non parallel act range
        tp_size: tensor parallel size
        model_type: model type
    '''
    layer_name_parts = layer_name.split(".")
    mlp_layer_name = ".".join(layer_name_parts[:-1])
    gate_weight, up_weight = merge_merged_weight(named_parameters, weight_name, tp_size)
    gate_up_name = layer_name_parts[-1]
    gate_up_list = smooth_model_config[model_type]["gate_up_list"]
    gate_up_list_len = len(gate_up_list)
    is_gate_up = smooth_model_config[model_type]["is_gate_up"]
    if gate_up_list_len == 2:
        gate_layer_name = f"{mlp_layer_name}.{gate_up_list[0]}"
        up_layer_name = f"{mlp_layer_name}.{gate_up_list[1]}"
    elif gate_up_list_len == 1:
        gate_up_layer_name = f"{mlp_layer_name}.{gate_up_list[0]}"

    if gate_up_list_len == 2:
        merged_act_range[gate_layer_name]["x"] = layer_range["x"]
        merged_act_range[up_layer_name]["x"] = layer_range["x"]
        merged_act_range[gate_layer_name]["is_merge"] = True
        merged_act_range[up_layer_name]["is_merge"] = True

        merged_named_parameters[f"{gate_layer_name}.weight"] = gate_weight
        merged_named_parameters[f"{up_layer_name}.weight"] = up_weight
    elif gate_up_list_len == 1:
        merged_act_range[gate_up_layer_name]["x"] = layer_range["x"]
        merged_gate_up_weight_list = [gate_weight, up_weight] if is_gate_up is True else [up_weight, gate_weight]
        merged_named_parameters[f"{gate_up_layer_name}.weight"] = torch.cat(merged_gate_up_weight_list, dim=0)

    if bias_name in named_parameters[0]:
        gate_bias, up_bias = merge_merged_weight(named_parameters, bias_name, tp_size)
        if gate_up_list_len == 2:
            merged_named_parameters[f"{gate_layer_name}.bias"] = gate_bias
            merged_named_parameters[f"{up_layer_name}.bias"] = up_bias
        elif gate_up_list_len == 1:
            merged_gate_up_bias_list = [gate_bias, up_bias] if is_gate_up is True else [up_bias, gate_bias]
            merged_named_parameters[f"{gate_up_layer_name}.bias"] = torch.cat(merged_gate_up_bias_list, dim=0)

    return gate_up_name


def convert_to_col_weight_except_qkv_merged(layer_name, weight_name, bias_name, named_parameters,
                                            merged_named_parameters, layer_range, merged_act_range, tp_size):
    '''
    convert colum parallel named parameters to non parallel named parameters
    args:
        layer_name: layer name
        weight_name: weight name
        bias_name: bias name
        named_parameters: parallel hugging face named parameters
        merged_named_parameters: non parallel hugging face named parameters
        layer_range: parallel layer range info
        merged_act_range: non parallel act range
        tp_size: tensor parallel size
    '''
    if layer_range['is_linear']:
        merged_act_range[layer_name]["x"] = layer_range["x"]
    merged_named_parameters[weight_name] = torch.cat(
        [named_parameters[tp_id][weight_name] for tp_id in range(0, tp_size)], dim=0)
    if bias_name in named_parameters[0]:
        merged_named_parameters[bias_name] = torch.cat(
            [named_parameters[tp_id][bias_name] for tp_id in range(0, tp_size)], dim=0)


def convert_to_row_weight(act_layer_name, act_range, layer_name, weight_name, bias_name, named_parameters,
                          merged_named_parameters, layer_range, merged_act_range, tp_size):
    '''
    convert row parallel named parameters to non parallel named parameters
    args:
        act_layer_name: act layer name
        act_range: parallel act_range
        layer_name: layer name
        weight_name: weight name
        bias_name: bias name
        named_parameters: parallel hugging face named parameters
        merged_named_parameters: non parallel hugging face named parameters
        layer_range: parallel layer range info
        merged_act_range: non parallel act range
        tp_size: tensor parallel size
    '''
    if layer_range['is_linear']:
        if isinstance(layer_range['x'], torch.Tensor):
            merged_act_range[layer_name]['x'] = torch.cat(
                [act_range[tp_id][act_layer_name]['x'] for tp_id in range(0, tp_size)], dim=0)
        else:
            merged_act_range[layer_name]['x'] = None

    merged_named_parameters[weight_name] = torch.cat(
        [named_parameters[tp_id][weight_name] for tp_id in range(0, tp_size)], dim=1)
    if bias_name in named_parameters[0]:
        merged_named_parameters[bias_name] = named_parameters[0][bias_name]


def convert_to_layer_merged(act_layer_name, act_range, layer_name, weight_name, bias_name, named_parameters,
                            merged_named_parameters, layer_range, merged_act_range, tp_size, args):
    '''
    convert parallel layer named parameters to non parallel layer named parameters
    args:
        act_layer_name: act layer name
        act_range: parallel act_range
        layer_name: layer name
        weight_name: weight name
        bias_name: bias name
        named_parameters: parallel hugging face named parameters
        merged_named_parameters: non parallel hugging face named parameters
        layer_range: parallel layer range info
        merged_act_range: non parallel act range
        tp_size: tensor parallel size
    '''
    qkv_name = "qkv_proj"
    gate_up_name = "gate_up_proj"

    if layer_range['split'] == 'col':  # col
        # merge weight
        if layer_range["is_qkv"]:
            qkv_name = convert_to_merged_qkv_weight(layer_name, weight_name, bias_name, named_parameters,
                                                    merged_named_parameters, layer_range, merged_act_range, tp_size,
                                                    args)

        elif layer_range["is_merge"]:
            gate_up_name = convert_to_merged_merged_weight(layer_name, weight_name, bias_name, named_parameters,
                                                           merged_named_parameters, layer_range, merged_act_range,
                                                           tp_size, args.model_type)
        else:
            convert_to_col_weight_except_qkv_merged(layer_name, weight_name, bias_name, named_parameters,
                                                    merged_named_parameters, layer_range, merged_act_range, tp_size)
    else:  # row
        convert_to_row_weight(act_layer_name, act_range, layer_name, weight_name, bias_name, named_parameters,
                              merged_named_parameters, layer_range, merged_act_range, tp_size)

    return qkv_name, gate_up_name


def get_mlp_layer_name_and_subfix(name):
    '''
    gate mlp_layer_name and weight subfix
    args:
        name: weight's name
    '''
    if name.endswith("weight"):
        layer_name = name[:-7]
        subfix = "weight"
    elif name.endswith("bias"):
        layer_name = name[:-5]
        subfix = "bias"
    else:
        layer_name = name
        subfix = "weight"
    layer_name_parts = layer_name.split(".")
    mlp_layer_name = ".".join(layer_name_parts[:-1])

    return mlp_layer_name, subfix


def convert_to_moe_gate_up_weight_merged(named_parameters, merged_named_parameters, merged_act_range, tp_size,
                                         experts_num, gate_up_name_list, is_merged, is_gate_up, model_type):
    '''
    convert moe's gate_up weight as unparallel weight
    args:
        named_parameters: parallel hugging face named parameters
        merged_named_parameters: non parallel hugging face named parameters
        merged_act_range: non parallel act range
        tp_size: tensor parallel size
        expert_num: expert numbers
        gate_up_name_list: gate_proj module name and up_proj model name list
        is_merged: is gate and up_proj merge together
        is_gate_up: gate first or up first when merged
        model_type: model type
    '''
    moe_list = smooth_model_config[model_type]["moe_list"]
    gate_up_list = moe_list["gate_up_list"]
    for name in gate_up_name_list:
        mlp_layer_name, subfix = get_mlp_layer_name_and_subfix(name)
        gate_proj = None
        if is_merged:
            gate_proj, up_proj = merge_merged_weight(named_parameters, name, tp_size, dim=1)
        else:
            up_proj = torch.cat([named_parameters[tp_id][name] for tp_id in range(0, tp_size)], dim=1)
        if gate_proj is not None:
            gate_layer_names = [f"{mlp_layer_name}.experts.{i}.{gate_up_list[1]}" for i in range(experts_num)]
            gate_split_tensors = torch.split(gate_proj, split_size_or_sections=1, dim=0)
            gate_proj_experts = [t.squeeze(0) for t in gate_split_tensors]
            for expert_id in range(experts_num):
                gate_layer_name = gate_layer_names[expert_id]
                merged_named_parameters[f"{gate_layer_name}.{subfix}"] = gate_proj_experts[expert_id]
                merged_act_range[gate_layer_name]["x"] = None

        up_layer_names = [f"{mlp_layer_name}.experts.{i}.{gate_up_list[2]}" for i in range(experts_num)]
        up_split_tensors = torch.split(up_proj, split_size_or_sections=1, dim=0)
        up_proj_experts = [t.squeeze(0) for t in up_split_tensors]
        for expert_id in range(experts_num):
            up_layer_name = up_layer_names[expert_id]
            merged_named_parameters[f"{up_layer_name}.{subfix}"] = up_proj_experts[expert_id]
            merged_act_range[up_layer_name]["x"] = None


def convert_to_moe_down_weight_merged(named_parameters, merged_named_parameters, merged_act_range, tp_size, experts_num,
                                      down_name_list, model_type):
    '''
    convert moe's down weight as unparallel weight
    args:
        named_parameters: parallel hugging face named parameters
        merged_named_parameters: non parallel hugging face named parameters
        merged_act_range: non parallel act range
        tp_size: tensor parallel size
        expert_num: expert numbers
        down_name_list: down_proj module name list
        model_type: model type
    '''
    moe_list = smooth_model_config[model_type]["moe_list"]
    down_list = moe_list["down_list"]
    for name in down_name_list:
        mlp_layer_name, subfix = get_mlp_layer_name_and_subfix(name)
        down_proj = torch.cat([named_parameters[tp_id][name] for tp_id in range(0, tp_size)], dim=-1)

        down_layer_names = [f"{mlp_layer_name}.experts.{i}.{down_list[1]}" for i in range(experts_num)]
        down_split_tensors = torch.split(down_proj, split_size_or_sections=1, dim=0)
        down_proj_experts = [t.squeeze(0) for t in down_split_tensors]
        for expert_id in range(experts_num):
            down_layer_name = down_layer_names[expert_id]
            merged_named_parameters[f"{down_layer_name}.{subfix}"] = down_proj_experts[expert_id]
            merged_act_range[down_layer_name]["x"] = None


def convert_to_moe_layer_merged(named_parameters, merged_named_parameters, merged_act_range, tp_size, model_type):
    '''
    convert moe weight as unparallel weight
    args:
        named_parameters: parallel hugging face named parameters
        merged_named_parameters: non parallel hugging face named parameters
        merged_act_range: non parallel act range
        tp_size: tensor parallel size
        model_type: model type
    '''
    moe_list = smooth_model_config[model_type]["moe_list"]
    gate_up_name = moe_list["gate_up_list"][0]
    down_name = moe_list["down_list"][0]
    is_merged = moe_list["is_merged"]
    is_gate_up = smooth_model_config[model_type]["is_gate_up"]

    gate_up_name_list = [name for name, _ in named_parameters[0].items() if gate_up_name in name]
    down_name_list = [name for name, _ in named_parameters[0].items() if down_name in name]
    len_gate_up_name_list = len(gate_up_name_list)
    len_down_name_list = len(down_name_list)
    assert len_gate_up_name_list > 0, f"len_gate_up_name_list({len_gate_up_name_list}) must be biger than 0"
    assert len_down_name_list > 0, f"len_down_name_list{len_down_name_list} must be biger than 0"
    param = named_parameters[0][gate_up_name_list[0]]
    experts_num = param.shape[0]

    convert_to_moe_gate_up_weight_merged(named_parameters, merged_named_parameters, merged_act_range, tp_size,
                                         experts_num, gate_up_name_list, is_merged, is_gate_up, model_type)
    convert_to_moe_down_weight_merged(named_parameters, merged_named_parameters, merged_act_range, tp_size, experts_num,
                                      down_name_list, model_type)

    return gate_up_name, down_name


def should_include(key, parameters, exclude_names):
    '''
    key shouldnot include in parameters and exlude_names
    args:
        parameters: named parameters
        exclude_names: excluded nameds list
    '''
    return key not in parameters and not any(exclude_name in key for exclude_name in exclude_names)


def valid_act_range(act_layer_name, layer_range):
    '''
    valid act_range, mainly filter inf and nan values in x field
    args:
        act_layer_name: act layer name
        layer_range: act layer value
    '''
    act_range_x = layer_range["x"]
    if act_range_x is not None and isinstance(act_range_x, torch.Tensor) and (torch.isinf(act_range_x).any()
                                                                              or torch.isnan(act_range_x).any()):
        mask = torch.isinf(act_range_x) | torch.isnan(act_range_x)
        act_range_x[mask] = 1e-6
        logger.warning(f"act_range_x in layer:{act_layer_name} has nan or inf values, force to 1e-6")


def convert_to_merged(act_range, named_parameters, tp_size, args):
    '''
    convert parallel act_range and named parameters to non parallel format.
    args:
        act_range: parallel act_range
        named_parameters: parallel named parameters
        tp_size: tensor parallel size
        args: argument
    '''
    model_type = args.model_type
    merged_act_range = defaultdict(lambda: {"x": None, "is_qkv": False, "is_merge": False,})
    merged_named_parameters = {}
    input_id_list = []
    if tp_size == 1:
        act_range = [act_range]
        named_parameters = [named_parameters]

    exclude_names = set()

    for act_layer_name, layer_range in act_range[0].items():
        valid_act_range(act_layer_name, layer_range)
        layer_name, weight_name, bias_name = get_layer_weight_bias_name(model_type, act_layer_name)
        qkv_name, gate_up_name = convert_to_layer_merged(act_layer_name, act_range, layer_name, weight_name, bias_name,
                                                         named_parameters, merged_named_parameters, layer_range,
                                                         merged_act_range, tp_size, args)
        exclude_names.update({qkv_name, gate_up_name})

        if layer_range['split'] == 'col' and layer_range["is_qkv"] and len(layer_range["input_id"]) > 0:
            input_id_list = layer_range["input_id"]

    if smooth_model_config[model_type]["moe_list"] is not None:
        w1_name, w2_name = convert_to_moe_layer_merged(named_parameters, merged_named_parameters, merged_act_range,
                                                       tp_size, model_type)
        exclude_names.update({w1_name, w2_name})

    merged_named_parameters.update({
        key: value
        for key, value in named_parameters[0].items()
        if should_include(key, merged_named_parameters, exclude_names)
    })

    modify_layer_weight_bias_name(model_type, merged_named_parameters)

    sorted_named_parameters = OrderedDict(sorted(merged_named_parameters.items(), key=lambda item: item[0]))
    sorted_merged_act_range = OrderedDict(sorted(merged_act_range.items(), key=lambda item: item[0]))

    return sorted_merged_act_range, sorted_named_parameters, input_id_list


def exclude_expert_in_merged_act_range(merged_act_range):
    '''
    delete expert info in merged_act_range.
    args:
        merged_act_range: non parallel act range
    '''
    quantized_act_range = defaultdict(lambda: {"x": None})
    for key, value in merged_act_range.items():
        if ".experts." not in key:
            quantized_act_range[key] = value

    return quantized_act_range


def copy_files_except_extensions(input_dir, output_dir, extensions):
    '''
    copy python files with extension in extensions from input_dir to output_dir, and keey sub directory is same
    args:
        input_dir: input directory
        output_dir: output directory
        extensions: the copy files extension
    '''
    # 遍历输入目录及其子目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not any(file.endswith(ext) for ext in extensions):
                # 构建源文件和目标文件的完整路径
                src_file = os.path.join(root, file)
                # 计算相对路径
                rel_path = os.path.relpath(root, input_dir)
                # 构建目标目录路径
                dst_dir = os.path.join(output_dir, rel_path)
                # 确保目标目录存在
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                dst_file = os.path.join(dst_dir, file)
                # 复制文件
                shutil.copy2(src_file, dst_file)
                logger.info(f'Copied {src_file} to {dst_file}')


def cleanup():
    '''
    cleanup memory resource
    '''
    gc.collect()
    torch.cuda.empty_cache()


def generate_datetime():
    '''
    generate current datetime
    '''
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_datetime
