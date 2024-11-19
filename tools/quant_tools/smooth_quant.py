import argparse
import torch
from datasets import load_dataset
import logging

from sinfer import LLM, SamplingParams

from utils_internal import convert_to_merged, exclude_expert_in_merged_act_range, cleanup

from input_context import prepare_inputs

from dump_smooth import save_prompt_token_ids, save_input_ids, save_act_range, save_weights, save_generate_weights

from model_special import smooth_model_config


logger = logging.getLogger(__name__)


def generate_prompts(args: argparse.Namespace):
    '''
    generate prompts
    '''
    if args.eval_task == 'code_completion':
        dataset_name = "openai_humaneval"
        dataset_revision = None
        dataset_input_key = 'prompt'
        dataset_split = 'train'
    elif args.eval_task == 'summarize':
        dataset_name = "ccdv/cnn_dailymail"
        dataset_revision = "3.0.0"
        dataset_input_key = 'article'
        dataset_split = 'train'
    elif args.eval_task == 'summarize_long':
        dataset_name = "tau/zero_scrolls"
        dataset_revision = 'squality'
        dataset_input_key = 'input'
        # only this split contains reference strings
        dataset_split = 'validation'
    elif args.eval_task == 'summarize_hg':
        dataset_name = "cnn_dailymail"
        dataset_revision = "3.0.0"
        dataset_input_key = 'article'
        dataset_split = 'validation'

    dataset = load_dataset(dataset_name,
                           dataset_revision,
                           cache_dir=args.dataset_cache_dir,
                           split=dataset_split,
                           trust_remote_code=True)
    num_samples = min(args.num_samples, len(dataset))
    prompts = dataset[0:num_samples][dataset_input_key]

    prompt_token_ids = []
    if args.has_prompt_token_id:
        batch_input_ids = prepare_inputs(prompts,
                                         args.tokenizer,
                                         args.model_name,
                                         args.model_version,
                                         args.max_input_length,
                                         eval_task=args.eval_task,
                                         add_special_tokens=args.add_special_tokens)
        save_prompt_token_ids(batch_input_ids, args)
        for i in range(num_samples):
            prompt_token_ids.append(batch_input_ids[i].tolist())

    if len(prompts) == 0:
        prompts = None

    if len(prompt_token_ids) == 0:
        prompt_token_ids = None

    return prompts, prompt_token_ids


@torch.no_grad()
def get_smooth_cal_weight(name, weight, name_parameters, act_range, model_type):
    '''
    get cal_weight for smooth process to solve q/k/v and gate/up layer merged condition in sinfer
    args:
        name: weight name
        weight: weight value
        name_parameters: named parameters
        act_range: layer act range info of name
        model_type: model type
    '''
    if act_range["is_qkv"] is True:
        name_parts = name.split(".")
        self_attn_layer_name = ".".join(name_parts[:-2])
        qkv_list = smooth_model_config[model_type]["qkv_list"]
        q_weight_name = f"{self_attn_layer_name}.{qkv_list[0]}.weight"
        k_weight_name = f"{self_attn_layer_name}.{qkv_list[1]}.weight"
        v_weight_name = f"{self_attn_layer_name}.{qkv_list[2]}.weight"
        q_weight = name_parameters[q_weight_name]
        k_weight = name_parameters[k_weight_name]
        v_weight = name_parameters[v_weight_name]
        cal_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    elif act_range["is_merge"] is True:
        name_parts = name.split(".")
        mlp_layer_name = ".".join(name_parts[:-2])
        gate_up_list = smooth_model_config[model_type]["gate_up_list"]
        gate_weight_name = f"{mlp_layer_name}.{gate_up_list[0]}.weight"
        up_weight_name = f"{mlp_layer_name}.{gate_up_list[1]}.weight"
        gate_weight = name_parameters[gate_weight_name]
        up_weight = name_parameters[up_weight_name]
        cal_weight = torch.cat([gate_weight, up_weight], dim=0)
    else:
        cal_weight = weight

    return cal_weight


@torch.no_grad()
def cal_smoother(weight, act_range_x, alpha=0.5):
    '''
    calculate smoother value
    args:
        weight: smoother weight
        act_range_x: activation max value of per channel
        alpha: smooth factor, default 0.5
    '''
    assert weight.shape[-1] == act_range_x.numel()
    weight_scales = weight.view(-1, weight.shape[-1])
    weight_scales = weight_scales.abs().max(dim=0)[0]
    weight_scales = weight_scales.to(float).clamp(min=1e-6)
    smoother = (act_range_x.to(weight_scales.device).to(float).pow(alpha) /
                weight_scales.pow(1 - alpha)).clamp(min=1e-6)

    return smoother


@torch.no_grad()
def cal_qweight_scales(sweight, smooth_act_range_x, per_token, per_channel):
    '''
    calculate quantized weight anc scales
    args:
        sweight: weight which has been divided by smoother value
        smooth_act_range_x: activation max value which has beed divide by smoother value
        per_token: bool, means whether calculate the weight and scales dynamically
        per_channel: bool, mean whether calculate the weight and scales by channel
    '''
    scale_x_quant_orig_t = smooth_act_range_x.max() / 127.0
    smooth_act_range_w = sweight.abs().max(dim=-1)[0]
    smooth_act_range_w = smooth_act_range_w.to(float).clamp(min=1e-6)
    scale_w_quant_orig_c = smooth_act_range_w / 127.0
    scale_w_quant_orig_t = smooth_act_range_w.max() / 127

    if per_channel:
        qweight = (sweight / scale_w_quant_orig_c[..., None])
    else:
        qweight = (sweight / scale_w_quant_orig_t)

    qweight = qweight.clip(-128, 127).to(torch.int8)

    scale_to_int = 1 / scale_x_quant_orig_t

    if per_token:
        if per_channel:
            per_channel_scale = scale_w_quant_orig_c
        else:
            per_channel_scale = scale_w_quant_orig_t
    else:
        if per_channel:
            per_channel_scale = scale_x_quant_orig_t * scale_w_quant_orig_c
            hidden_size = smooth_act_range_x.numel()
            scale_to_int = scale_to_int.repeat(hidden_size)
        else:
            per_channel_scale = scale_x_quant_orig_t * scale_w_quant_orig_t

    per_channel_scale = per_channel_scale.squeeze()
    if per_channel_scale.numel() == 1 and per_channel_scale.dim() == 0:
        per_channel_scale = per_channel_scale.unsqueeze(0)

    if scale_to_int.numel() == 1 and scale_to_int.dim() == 0:
        scale_to_int = scale_to_int.unsqueeze(0)

    sinfo = [
        scale_w_quant_orig_t.item(), scale_x_quant_orig_t.item(),
        scale_w_quant_orig_t.item() / scale_x_quant_orig_t.item()
    ]
    return qweight, per_channel_scale, scale_to_int, sinfo


def check_smooth_weight_vaild(name, qweight, per_channel_scale, smooth, qzeros, scale_to_int):
    '''
    check whether nan/inf appears in qweight, per_channel_scale, smooth, qzeros, scale_to_int
    '''
    if torch.isinf(qweight).any() or torch.isnan(qweight).any():
        logger.error(f"name:{name} qweight has inf or nan")
    if torch.isinf(per_channel_scale).any() or torch.isnan(per_channel_scale).any():
        logger.error(f"name:{name} per_channel_scale has inf or nan")
    if torch.isinf(smooth).any() or torch.isnan(smooth).any():
        logger.error(f"name:{name} smooth has inf or nan")
    if torch.isinf(scale_to_int).any() or torch.isnan(scale_to_int).any():
        logger.error(f"name:{name} scale_to_int has inf or nan")
    if qzeros is not None and (torch.isinf(qzeros).any() or torch.isnan(qzeros).any()):
        logger.error(f"name:{name} qzeros has inf or nan")


@torch.no_grad()
def cal_smooth_weight(name, act_range_x, weight, smooth_value, has_qzeros, per_token, per_channel, cal_weight):
    '''
    calculate qweight, scales, smooth, qzeros
    args:
        name: weight name
        act_range_x: activation max value of per channel
        weight: weight to be quantized
        smooth_value: smooth value
        has_qzeros: which generate qzeros weight
        per_token: bool, means whether calculate the weight and scales dynamically
        per_channel: bool, mean whether calculate the weight and scales by channel
        model_type: model type
    '''
    smoother = cal_smoother(cal_weight, act_range_x, smooth_value)
    smooth_act_range_x = act_range_x / smoother
    sweight = weight * (smoother.view(1, -1))
    qweight, per_channel_scale, scale_to_int, sinfo = cal_qweight_scales(sweight, smooth_act_range_x, per_token,
                                                                         per_channel)
    qweight = qweight.reshape(weight.shape)
    smooth = 1 / smoother
    smooth = smooth.squeeze()
    if has_qzeros:
        qzeros = torch.zeros_like(per_channel_scale, dtype=torch.int32)
    else:
        qzeros = None

    # check_smooth_weight_vaild(name, qweight, per_channel_scale, smooth, qzeros, scale_to_int)

    return qweight, per_channel_scale, smooth, qzeros, scale_to_int, sinfo


@torch.no_grad()
def generate_smooth_weight(act_range, name_parameters, args):
    '''
    generate smooth weight
    args:
        act_range: act_range collected in model running
        name_parameters: hugging face model named parameters
        args: argument from main
    '''
    smooth_weight = {}
    smooth_info = {}
    has_qzeros = args.has_qzeros
    smooth_value = args.smooth_value

    smooth_info["title"] = ["max_scale_w, max_scale_x, max_scale_w/max_scale_x"]

    for name, param in name_parameters.items():
        if name.endswith("bias"):
            smooth_weight[name] = param
            continue
        name_parts = name.split(".")
        layer_name = ".".join(name_parts[:-1])
        if layer_name in act_range:
            act_range_x = act_range[layer_name]['x']
            cal_weight = get_smooth_cal_weight(name, param, name_parameters, act_range[layer_name], args.model_type)
            qweight, per_channel_scale, smooth, qzeros, scale_to_int, sinfo = cal_smooth_weight(
                name, act_range_x, param, smooth_value, has_qzeros, args.per_token, args.per_channel, cal_weight)

            per_channel_scale = per_channel_scale.to(args.torch_scales_smooth_dtype)
            smooth = smooth.to(args.torch_scales_smooth_dtype)
            scale_to_int = scale_to_int.to(args.torch_scales_smooth_dtype)

            smooth_weight[f'{layer_name}.qweight'] = qweight
            smooth_weight[f'{layer_name}.per_channel_scale'] = per_channel_scale

            if args.per_token is True:
                smooth_weight[f'{layer_name}.smooth'] = smooth
            else:
                scale_to_int = scale_to_int * smooth
                smooth_weight[f'{layer_name}.scale_to_int'] = scale_to_int

            if has_qzeros:
                smooth_weight[f'{layer_name}.qzeros'] = qzeros

            smooth_info[name] = sinfo
        else:
            smooth_weight[name] = param

    return smooth_weight, smooth_info


def generate_weights_of_smoothquant(llm: LLM, args: argparse.Namespace):
    '''
    generate smoothquant weights
    args:
        llm: LLM instance
        args: argument from main
    '''
    prompts, prompt_token_ids = generate_prompts(args)

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=args.output_len,
                                     repetition_penalty=args.repetition_penalty,
                                     temperature=args.temperature,
                                     top_p=args.top_p,
                                     top_k=args.top_k,
                                     length_penalty=args.length_penalty)

    tp_size = args.tp_size

    if tp_size > 1:
        llm.llm_engine.model_executor._run_workers("setup_smooth_hook", args.dump_input_ids)
    else:
        llm.llm_engine.model_executor.driver_worker.setup_smooth_hook(args.dump_input_ids)

    llm.generate(prompts, sampling_params, prompt_token_ids=prompt_token_ids, use_tqdm=True)

    logger.info("llm generate finished")

    if tp_size > 1:
        llm.llm_engine.model_executor._run_workers("remove_hooks")
        act_range = llm.llm_engine.model_executor._run_workers("get_act_range")
        named_parameters = llm.llm_engine.model_executor._run_workers("get_named_parameters")
    else:
        llm.llm_engine.model_executor.driver_worker.remove_hooks()
        act_range = llm.llm_engine.model_executor.driver_worker.get_act_range()
        named_parameters = llm.llm_engine.model_executor.driver_worker.get_named_parameters()

    del llm
    del prompts
    del prompt_token_ids
    cleanup()

    logger.info("get act_range and named_parameters from llm finished")

    merged_act_range, merged_named_parameters, input_id_list = convert_to_merged(act_range, named_parameters, tp_size,
                                                                                 args)

    save_input_ids(input_id_list, args)
    save_act_range(merged_act_range, args)
    save_weights(merged_named_parameters, args)

    del act_range
    del named_parameters
    cleanup()

    logger.info("get merged_act_range and merged_named_parameters finished")

    # currently we can't collect export's X info, so expert's weight isn't quantized now
    merged_act_range = exclude_expert_in_merged_act_range(merged_act_range)

    smooth_weight, smooth_info = generate_smooth_weight(merged_act_range, merged_named_parameters, args)
    save_generate_weights(smooth_weight, args)

    del merged_act_range
    del merged_named_parameters
    cleanup()

    logger.info("get smooth_weight finished")

    return smooth_weight, smooth_info
