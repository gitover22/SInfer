import re

# model_type, qkv_list, gate_up_list, is_gate_up
smooth_model_config = {
    "llama": {
        "qkv_list": ["q_proj", "k_proj", "v_proj"],
        "gate_up_list": ["gate_proj", "up_proj"],
        "is_gate_up": True,
        "moe_list": None
    },
    "qwen2": {
        "qkv_list": ["q_proj", "k_proj", "v_proj"],
        "gate_up_list": ["gate_proj", "up_proj"],
        "is_gate_up": True,
        "moe_list": None
    },
    "qwen": {
        "qkv_list": ["c_attn"],
        "gate_up_list": ["w2", "w1"],
        "is_gate_up": True,
        "moe_list": None
    },
    "baichuan": {
        "qkv_list": ["W_pack"],
        "gate_up_list": ["gate_proj", "up_proj"],
        "is_gate_up": True,
        "moe_list": None
    },
    "chatglm": {
        "qkv_list": ["query_key_value"],
        "gate_up_list": ["dense_h_to_4h"],
        "is_gate_up": True,
        "moe_list": None
    },
    "gpt_neox": {
        "qkv_list": ["query_key_value"],
        "gate_up_list": [],
        "is_gate_up": True,
        "moe_list": None
    },
    "mixtral": {
        "qkv_list": ["q_proj", "k_proj", "v_proj"],
        "gate_up_list": ["w1", "w3"],
        "is_gate_up": True,
        "moe_list": {
            "gate_up_list": ["block_sparse_moe.w13", "w1", "w3"],
            "down_list": ["block_sparse_moe.w2", "w2"],
            "is_merged": True
        }
    },
    "qwen2_moe": {
        "qkv_list": ["q_proj", "k_proj", "v_proj"],
        "gate_up_list": ["gate_proj", "up_proj"],
        "is_gate_up": True,
        "moe_list": {
            "gate_up_list": ["mlp.w1", "gate_proj", "up_proj"],
            "down_list": ["mlp.w2", "down_proj"],
            "is_merged": True
        }
    },
    "falcon": {
        "qkv_list": ["query_key_value"],
        "gate_up_list": ["dense_h_to_4h"],
        "is_gate_up": True,
        "moe_list": None
    },
}


def get_layer_weight_bias_name(model_type, layer_name):
    '''
    Specially adjust the condition that layer_name and weight/bias name are different,
    or the condithon that weight/bias name is not {layer_name}.weight/bias
    '''
    weight_name = None
    bias_name = None
    # chatglm transformer.output_layer.weight == lm_head_weight
    if model_type == "chatglm" and "output_layer" in layer_name:
        layer_name = "lm_head"
        weight_name = f"{layer_name}_weight"
        bias_name = f"{layer_name}_bias"

    if model_type == "falcon" and "word_embeddings" in layer_name:
        layer_name = "lm_head"
        weight_name = f"{layer_name}_weight"
        bias_name = f"{layer_name}_bias"

    if model_type == "qwen2" and "lm_head" == layer_name:
        weight_name = "lm_head_weight"
        bias_name = "lm_head_bias"

    if weight_name is None:
        weight_name = f"{layer_name}.weight"
    if bias_name is None:
        bias_name = f"{layer_name}.bias"

    return layer_name, weight_name, bias_name


def modify_layer_weight_bias_name(model_type, named_parameters):
    '''
    modify special condition that sinfer layer_name isn't same as hf layer name
    '''
    # Mapping for model type specific adjustments
    mapping = {
        "chatglm": {
            "lm_head_weight": "transformer.output_layer.weight",
            "transformer.embedding.weight": "transformer.embedding.word_embeddings.weight"
        },
        "falcon": {
            "lm_head_weight": "transformer.word_embeddings.weight"
        }
    }

    if model_type in mapping:
        for old_key, new_key in mapping[model_type].items():
            if old_key in named_parameters:
                named_parameters[new_key] = named_parameters.pop(old_key)


def extract_numbers(string):
    '''
    extract a string to number
    '''
    # 使用正则表达式找到字符串中的所有数字部分
    matches = re.findall(r'\d+', string)

    # 将所有匹配的数字部分转换为整数
    numbers = [int(match) for match in matches]

    return numbers[-1] if len(numbers) > 0 else 0


def get_qkv_distribution(model_type, model_version, hf_config):
    '''
    Get qkv distribution: n3sh or 3nsh
    n3sh: [head_num, 3, head_size, hidden_size]
    3nsh: [3, head_num, head_size, hidden_size]
    sinfer default qkv distribution is 3nsh, so here need to provide n3sh model info, tools will convert 3nsh to n3sh
    to be same as hugging face qkv distribution
    This is only for packge qkv layer and it's distribution is n3sh
    '''
    is_n3sh = False
    head_num = 0
    kv_head_num = 0
    if (model_type == "chatglm" and extract_numbers(model_version) == 0) or model_type in ["bloom", "gpt_neox"]:
        is_n3sh = True
        head_num = hf_config.num_attention_heads

        kv_head_num = head_num
    if model_type == "falcon":
        is_n3sh = True
        head_num = hf_config.num_attention_heads
        if hf_config.new_decoder_architecture:
            kv_head_num = hf_config.num_kv_heads
        elif hf_config.multi_query:
            kv_head_num = 1
        else:
            kv_head_num = head_num

    return is_n3sh, head_num, kv_head_num
