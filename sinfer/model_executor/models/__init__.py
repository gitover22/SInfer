import importlib
from typing import Dict, List, Optional, Type

import torch.nn as nn

from sinfer.logger import init_logger

logger = init_logger(__name__)

# Architecture -> (module, class).
_MODELS = {
    "AquilaModel": ("llama", "LlamaForCausalLM"),
    "AquilaForCausalLM": ("llama", "LlamaForCausalLM"),  # AquilaChat2
    "BaiChuanForCausalLM": ("baichuan", "BaiChuanForCausalLM"),  # baichuan-7b
    "BaichuanForCausalLM": ("baichuan", "BaichuanForCausalLM"),  # baichuan-13b
    "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    "CohereForCausalLM": ("commandr", "CohereForCausalLM"),
    "DbrxForCausalLM": ("dbrx", "DbrxForCausalLM"),
    "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    "DeepseekForCausalLM": ("deepseek", "DeepseekForCausalLM"),
    "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
    "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "GPT2LMHeadModel"),
    "GPTBigCodeForCausalLM": ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    "GPTJForCausalLM": ("gpt_j", "GPTJForCausalLM"),
    "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXForCausalLM"),
    "InternLMForCausalLM": ("llama", "LlamaForCausalLM"),
    "InternLM2ForCausalLM": ("internlm2", "InternLM2ForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "LlavaForConditionalGeneration":
    ("llava", "LlavaForConditionalGeneration"),
    # For decapoda-research/llama-*
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    "QuantMixtralForCausalLM": ("mixtral_quant", "MixtralForCausalLM"),
    # transformers's mpt class has lower case
    "MptForCausalLM": ("mpt", "MPTForCausalLM"),
    "MPTForCausalLM": ("mpt", "MPTForCausalLM"),
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMForCausalLM"),
    "OlmoForCausalLM": ("olmo", "OlmoForCausalLM"),
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "OrionForCausalLM": ("orion", "OrionForCausalLM"),
    "PhiForCausalLM": ("phi", "PhiForCausalLM"),
    "Phi3ForCausalLM": ("llama", "LlamaForCausalLM"),
    "QWenLMHeadModel": ("qwen", "QWenLMHeadModel"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "RWForCausalLM": ("falcon", "FalconForCausalLM"),
    "StableLMEpochForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
    "XverseForCausalLM": ("xverse", "XverseForCausalLM"),
    "FM9GForCausalLM": ("fm9g", "FM9GForCausalLM"), # 九格大模型
}

# Architecture -> type.
# out of tree models
_OOT_MODELS: Dict[str, Type[nn.Module]] = {}

# # Models not supported by ROCm.
# _ROCM_UNSUPPORTED_MODELS = []

# Models partially supported by ROCm.
# Architecture -> Reason.
# _ROCM_PARTIALLY_SUPPORTED_MODELS = {
#     "Qwen2ForCausalLM":
#     "Sliding window attention is not yet supported in ROCm's flash attention",
#     "MistralForCausalLM":
#     "Sliding window attention is not yet supported in ROCm's flash attention",
#     "MixtralForCausalLM":
#     "Sliding window attention is not yet supported in ROCm's flash attention",
# }


class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch in _OOT_MODELS:
            return _OOT_MODELS[model_arch]
        if model_arch not in _MODELS:
            return None

        module_name, model_cls_name = _MODELS[model_arch] # llama, LlamaForCausalLM
        module = importlib.import_module(
            f"sinfer.model_executor.models.{module_name}") # 导入对应的py文件，llama.py
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())

    @staticmethod
    def register_model(model_arch: str, model_cls: Type[nn.Module]):
        if model_arch in _MODELS:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.", model_arch,
                model_cls.__name__)
        global _OOT_MODELS
        _OOT_MODELS[model_arch] = model_cls


__all__ = [
    "ModelRegistry",
]
