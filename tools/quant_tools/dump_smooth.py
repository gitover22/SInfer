import torch
import os
import logging

logger = logging.getLogger(__name__)


def tensor_shape_to_string(tensor):
    '''
    convert a tensor shape to string description
    '''
    int_list = list(tensor.shape)
    str_list = [str(num) for num in int_list]
    str_shape = "x".join(str_list)
    return str_shape


def save_prompt_token_ids(prompt_input_ids, args):
    '''
    save prompt_token_id
    Args:
        prompt_input_ids: prompt input_id assiged to llm.generate
        args: arguments from main
    '''
    if args.dump_prompt_token_ids is not True:
        return
    output_dir = os.path.join(args.output_dir, "prompt_input_ids")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_len = len(prompt_input_ids)
    for data_index in range(data_len):
        tensor = prompt_input_ids[data_index]
        str_shape = tensor_shape_to_string(tensor)
        file_path = os.path.join(output_dir, f"prompt_input_ids_{data_index}_{str_shape}.pt")
        torch.save(tensor, file_path)
        logger.info(f"Saved input_ids[{data_index}] to {file_path}")


def save_input_ids(input_ids, args):
    '''
    save input_ids
    Args:
        input_ids: input of qkv with layer0
        args: arguments from main
    '''
    id_len = len(input_ids)
    if args.dump_input_ids is not True or id_len == 0:
        return
    output_dir = os.path.join(args.output_dir, "input_ids")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data_index in range(id_len):
        tensor = input_ids[data_index]
        str_shape = tensor_shape_to_string(tensor)
        file_path = os.path.join(output_dir, f"input_ids_{data_index}_{str_shape}.pt")
        torch.save(tensor, file_path)
        logger.info(f"Saved input_ids[{data_index}] to {file_path}")


def save_act_range(act_range, args):
    '''
    save act_range
    Args:
        act_range: save act_range collected when model running
        args: arguments from main
    '''
    if args.dump_act_range is not True:
        return
    output_dir = os.path.join(args.output_dir, "act_range")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for layer_name, layer_scale in act_range.items():
        for tensor_key, tensor_value in layer_scale.items():
            if isinstance(tensor_value, torch.Tensor):
                str_shape = tensor_shape_to_string(tensor_value)
                file_name = f'{layer_name}_{tensor_key}_{str_shape}.pt'
                file_path = os.path.join(output_dir, file_name)
                torch.save(tensor_value, file_path)
                logger.info(f"Saved act_range[{layer_name}][{tensor_key}] to {file_path}")


def save_weights(weights, args):
    '''
    save hugging face weights
    Args:
        weights: hugging face weights merged with llm model named parameters
        args: arguments from main
    '''
    if args.dump_weights is not True:
        return
    output_dir = os.path.join(args.output_dir, "weights")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for tensor_key, tensor_value in weights.items():
        str_shape = tensor_shape_to_string(tensor_value)
        file_name = f'{tensor_key}_{str_shape}.pt'
        file_path = os.path.join(output_dir, file_name)
        torch.save(tensor_value, file_path)
        logger.info(f"Saved weights[{tensor_key}] to {file_path}")


def save_generate_weights(weights, args):
    '''
    save quantizated weights
    Args:
        weights: quantized weights of smoothquant or weightonly
        args: arguments from main
    '''
    if args.dump_generate_weights is not True:
        return
    output_dir = os.path.join(args.output_dir, "generate_weights")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for tensor_key, tensor_value in weights.items():
        str_shape = tensor_shape_to_string(tensor_value)
        file_name = f'{tensor_key}_{str_shape}.pt'
        file_path = os.path.join(output_dir, file_name)
        torch.save(tensor_value, file_path)
        logger.info(f"Saved generate weights[{tensor_key}] to {file_path}")


def dump_save_x_y(name, x, y, index):
    '''
    dump x, y when inferrence
    output_dir need to modify by your self
    '''
    output_dir = "output_dir"
    x_output_dir = os.path.join(output_dir, "x_tensor")
    y_output_dir = os.path.join(output_dir, "y_tensor")
    if not os.path.exists(x_output_dir):
        os.makedirs(x_output_dir)
    if not os.path.exists(y_output_dir):
        os.makedirs(y_output_dir)

    x_file_name = os.path.join(x_output_dir, f"{name}_x_{index}.pt")
    y_file_name = os.path.join(y_output_dir, f"{name}_y_{index}.pt")
    if isinstance(x, tuple):
        x = x[0]
    if not os.path.exists(x_file_name):
        torch.save(x.cpu(), x_file_name)
    if not os.path.exists(y_file_name):
        torch.save(y.cpu(), y_file_name)
