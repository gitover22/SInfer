import os
import argparse
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, GenerationConfig)

from sinfer.transformers_utils.config import get_config
from utils_internal import (read_model_name, torch_dtype_to_str, str_dtype_to_torch)
from dump_smooth import save_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_dir', type=str, default=None)
    parser.add_argument('--output_dir',
                        type=str,
                        default="output_dir",
                        help="The path to save the quantized checkpoint")
    parser.add_argument('--model_version',
                        type=str,
                        default=None,
                        help="Set model version to replace parsing from _name_or_path in hf config.")
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        help="Set model type to replace parsing from model_type in hf config."
                        "if set is None and parsed also None, then set as model_version")
    parser.add_argument('--dtype',
                        type=str,
                        choices=['auto', 'float32', 'float16', 'bfloat16'],
                        default='auto',
                        help="if auto, use unquantized weight torch_dtype in config.json, else use setted dtype")
    parser.add_argument(
        '--dump_weights',
        action="store_true",
        default=True,
        help='dump weights of the converted model',
    )

    args = parser.parse_args()

    assert args.hf_model_dir, "Please set model_dir by --model_dir or --hf_model_dir"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.model_name, args.model_version, args.model_family, args.model_type = read_model_name(
        args.hf_model_dir, args.model_version, args.model_type)

    args.hf_config = get_config(args.hf_model_dir, trust_remote_code=True)

    if args.dtype == "auto":
        args.dtype = torch_dtype_to_str(args.hf_config.torch_dtype)

    args.torch_dtype = str_dtype_to_torch(args.dtype)
    args.hf_config.torch_dtype = args.torch_dtype

    if args.model_name == 'ChatGLMForCausalLM' and args.model_version == 'glm':
        auto_model_cls = AutoModelForSeq2SeqLM
    elif args.model_name == 'ChatGLMForCausalLM' and args.model_version == 'chatglm':
        auto_model_cls = AutoModel
    else:
        auto_model_cls = AutoModelForCausalLM
    model = auto_model_cls.from_pretrained(
        args.hf_model_dir,
        trust_remote_code=True,
        torch_dtype=args.torch_dtype)

    named_parameters = dict(model.named_parameters())
    save_weights(named_parameters, args)
