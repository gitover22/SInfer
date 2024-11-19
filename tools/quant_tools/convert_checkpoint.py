import argparse
import os

import sys
import time
import safetensors
import logging
import json

from sinfer import LLM
from sinfer.transformers_utils.config import get_config
from transformers.modeling_utils import shard_checkpoint

from smooth_quant import generate_weights_of_smoothquant
from weight_only import generate_weights_of_weight_only
from utils_internal import (read_model_name, load_tokenizer, torch_dtype_to_str, str_dtype_to_torch,
                            copy_files_except_extensions, generate_datetime)
from model_special import smooth_model_config

sys.path.append(os.getcwd())

logger = logging.getLogger("smooth_convert")


def main(args):
    '''
    main quantization logic
    '''
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=args.log_level,
        force=True,
    )

    tik = time.time()

    # Create an LLM.
    llm = LLM(model=args.hf_model_dir,
              tokenizer=args.tokenizer_dir,
              tensor_parallel_size=args.tp_size,
              dtype=args.dtype,
              enforce_eager=args.enforce_eager,
              trust_remote_code=True,
              block_size=args.block_size)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))

    logger.info(f'Load SInfer model takes: {t}')

    quantize_config = {}
    if args.use_weight_only:
        st_prefix = f"weight_{args.weight_only_precision}"
        quantized_weights = generate_weights_of_weight_only(llm, args)
        quantize_config['bits'] = 8 if args.weight_only_precision == "int8" else 4
        quantize_config['quant_method'] = "weightonly"
        quantize_config['quant_mode'] = "WeightOnly"

    if args.use_smoothquant:
        st_prefix = f"smoothquant_{args.smooth_value}"
        quantized_weights, smooth_info = generate_weights_of_smoothquant(llm, args)
        quantize_config['bits'] = 8
        quantize_config['quant_method'] = "smoothquant"
        quantize_config['quant_mode'] = "SmoothQuant"
        quantize_config['input_quant_method'] = "per_token" if args.per_token else "per_tensor"
        with open(os.path.join(args.output_dir, 'smooth_info.json'), 'w') as f:
            json.dump(smooth_info, f, indent=4)

    weight_name = f'{st_prefix}.safetensors'
    if args.max_shard_size is not None:
        shards, _ = shard_checkpoint(quantized_weights, max_shard_size=args.max_shard_size, weights_name=weight_name)
        for shard_name, shard_weight in shards.items():
            safetensors.torch.save_file(shard_weight, os.path.join(args.output_dir, shard_name))
    else:
        safetensors.torch.save_file(quantized_weights, os.path.join(args.output_dir, weight_name))
    logger.info(f'save_file: {st_prefix}.safetensors success')

    with open(os.path.join(args.output_dir, 'quantize_config.json'), 'w') as f:
        json.dump(quantize_config, f, indent=4)

    extensions = ['.bin', '.safetensors']
    copy_files_except_extensions(args.hf_model_dir, args.output_dir, extensions)

    args.hf_config.quantization_config = quantize_config
    args.hf_config.generate_datetime = generate_datetime()
    args.hf_config.save_pretrained(args.output_dir)

    logger.info(f'quantized {args.hf_model_dir} finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_dir', type=str, default=None)
    parser.add_argument('--tokenizer_dir',
                        default=None,
                        help='tokenizer path; defaults to hf_model_dir if left unspecified')
    parser.add_argument(
        '--enforce_eager',
        action="store_true",
        default=True,
        help='Whether to enforce eager execution. If True, we will disable CUDA graph and always execute the model '
        'in eager mode. If False, we will use CUDA graph and eager execution in hybrid.')
    parser.add_argument('--dtype',
                        type=str,
                        choices=['auto', 'float32', 'float16', 'bfloat16'],
                        default='auto',
                        help="if auto, use unquantized weight torch_dtype in config.json, else use setted dtype")
    parser.add_argument('--scales_smooth_dtype',
                        type=str,
                        choices=['auto', 'float32', 'float16', 'bfloat16'],
                        default='auto',
                        help="if auto, scales and smooth weights use args.dtype, else use the setted dtype")
    parser.add_argument('--eval_task',
                        type=str,
                        default='summarize',
                        choices=['summarize', 'summarize_long', 'code_completion', 'summarize_hg'])
    parser.add_argument("--dataset_cache_dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
    parser.add_argument('--log_level', type=int, default=logging.INFO)
    parser.add_argument('--num_samples', type=int, default=512, help='num prompt sample')
    parser.add_argument('--output_len',
                        type=int,
                        default=100,
                        help="Number of output sequences to return for the given prompt")
    parser.add_argument('--max_input_length',
                        type=int,
                        default=512,
                        help='max input length of the prompt, only valid when --has_prompt_token_id is setted')
    parser.add_argument('--block_size', type=int, default=-1, help='Token block size for contiguous chunks of tokens.')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--output_dir',
                        type=str,
                        default="output_dir",
                        help="The path to save the quantized checkpoint")
    parser.add_argument('--max_shard_size',
                        type=str,
                        default=None,
                        help="The maximum size of each sub-checkpoint. needs to be digits followed by a unit,"
                        " such as 5MB, 5GB and so on")
    parser.add_argument('--tp_size', type=int, default=1, help='N-way tensor parallelism size')
    parser.add_argument('--pp_size', type=int, default=1, help='N-way pipeline parallelism size, now supported num')
    parser.add_argument('--use_smoothquant',
                        default=False,
                        action="store_true",
                        help='Apply smoothquant to generate weight')
    parser.add_argument("--smooth_value",
                        type=float,
                        default=0.5,
                        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
                        " to Smoothquant the model, and output int8 weights."
                        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument('--per_channel',
                        action="store_true",
                        default=False,
                        help='By default, we use a single static scaling factor for the GEMM\'s result. '
                        'per_channel instead uses a different static scaling factor for each channel. '
                        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        action="store_true",
        default=False,
        help='By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument('--use_weight_only',
                        default=False,
                        action="store_true",
                        help='Quantize weights for the various GEMMs to INT4/INT8.'
                        'See --weight_only_precision to set the precision')
    parser.add_argument('--weight_only_precision',
                        const='int8',
                        type=str,
                        nargs='?',
                        default='int8',
                        choices=['int8', 'int4'],
                        help='Define the precision for the weights when using weight-only quantization.'
                        'You must also use --use_weight_only for that argument to have an impact.')
    parser.add_argument(
        '--has_qzeros',
        action="store_true",
        default=False,
        help='whether to add qzeros weight to sinfer_mlu weight',
    )
    parser.add_argument('--model_version',
                        type=str,
                        default=None,
                        help="Set model version to replace parsing from _name_or_path in hf config.")
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        help="Set model type to replace parsing from model_type in hf config."
                        "if set is None and parsed also None, then set as model_version")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument(
        '--has_prompt_token_id',
        action="store_true",
        default=False,
        help='whether to give llm.generate prompt_token_id',
    )
    parser.add_argument(
        '--enable_quantize_expert',
        action="store_true",
        default=False,
        help='enable quantize expert weight',
    )
    parser.add_argument(
        '--dump_prompt_token_ids',
        action="store_true",
        default=False,
        help='dump prompt_token_ids used by llm.generate ',
    )
    parser.add_argument(
        '--dump_input_ids',
        action="store_true",
        default=False,
        help='dump sinfer qkv used token ids at llm running',
    )
    parser.add_argument(
        '--dump_act_range',
        action="store_true",
        default=False,
        help='dump act range which is the max hidden dim value of input, output, weigth',
    )
    parser.add_argument(
        '--dump_weights',
        action="store_true",
        default=False,
        help='dump weights of the converted model',
    )
    parser.add_argument(
        '--dump_generate_weights',
        action="store_true",
        default=False,
        help='dump generate weights of the converted model',
    )

    args = parser.parse_args()

    assert args.hf_model_dir, "Please set model_dir by --model_dir or --hf_model_dir"
    assert args.pp_size == 1, "Pipeline parallelism is not supported."

    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir

    if args.has_prompt_token_id is False:
        args.dump_prompt_token_ids = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.model_name, args.model_version, args.model_family, args.model_type = read_model_name(
        args.hf_model_dir, args.model_version, args.model_type)
    assert args.model_type in smooth_model_config, f'''{args.model_type} hasn't supported,
      please add it's infomation in model_special.py by your self'''

    args.hf_config = get_config(args.hf_model_dir, trust_remote_code=True)

    if args.dtype == "auto":
        args.dtype = torch_dtype_to_str(args.hf_config.torch_dtype)

    if args.scales_smooth_dtype == "auto":
        args.scales_smooth_dtype = args.dtype

    args.torch_dtype = str_dtype_to_torch(args.dtype)
    args.torch_scales_smooth_dtype = str_dtype_to_torch(args.scales_smooth_dtype)
    args.hf_config.torch_dtype = args.torch_dtype

    args.tokenizer, args.pad_id, args.end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        model_name=args.model_name,
        model_version=args.model_version,
    )

    tik = time.time()
    main(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of converting checkpoints: {t}')
