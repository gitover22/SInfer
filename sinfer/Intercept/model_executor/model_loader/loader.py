
import torch
from torch import nn
from typing import List, Tuple, Generator, Optional
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.model_executor.model_loader.loader import DefaultModelLoader
from sinfer.model_executor.model_loader.loader import DummyModelLoader, _initialize_model
from sinfer.model_executor.model_loader.weight_utils import (np_cache_weights_iterator, initialize_dummy_weights)
from sinfer.config import LoadFormat
from safetensors.torch import safe_open
from sinfer.config import (DeviceConfig, LoadConfig, LoadFormat, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from sinfer.model_executor.model_loader.utils import set_default_torch_dtype


CAST_BFLOAT16_TO_FLOAT16_ENABLE = (torch.cuda.get_device_capability()[0] == 3)


def intercept_safetensors_weights_iterator(
    hf_weights_files: List[str]
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """迭代模型中的 safetensor 权重文件"""
    for st_file in hf_weights_files:  # 遍历所有 safetensor 权重文件
        with safe_open(st_file, framework="pt") as f: # 打开 safetensor 文件，指定使用 PyTorch 框架
            for name in f.keys():  # 遍历文件中的所有张量名称
                param = f.get_tensor(name) # 根据名称获取对应的张量
                # cast bfloat16 to float16 for MLU3xx
                if CAST_BFLOAT16_TO_FLOAT16_ENABLE and param.dtype == torch.bfloat16:
                    param = param.to(torch.float16) 
                yield name, param


def intercept_pt_weights_iterator(
    hf_weights_files: List[str]
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    for bin_file in hf_weights_files:
        state = torch.load(bin_file, map_location="cpu")
        for name, param in state.items():
            # cast bfloat16 to float16 for MLU3xx
            if CAST_BFLOAT16_TO_FLOAT16_ENABLE and param.dtype == torch.bfloat16:
                param = param.to(torch.float16)
            yield name, param
        del state
        torch.cuda.empty_cache()


def intercept_get_weights_iterator(
    self, model_name_or_path: str, revision: Optional[str],
    fall_back_to_pt: bool
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """根据加载格式获取模型权重的迭代器"""
    hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
        model_name_or_path, revision, fall_back_to_pt)
    if self.load_config.load_format == LoadFormat.NPCACHE:
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False
        return np_cache_weights_iterator(model_name_or_path,
                                            self.load_config.download_dir,
                                            hf_folder, hf_weights_files)
    # cast bfloat16 to float16 for MLU3xx
    if use_safetensors:
        return intercept_safetensors_weights_iterator(hf_weights_files)
    return intercept_pt_weights_iterator(hf_weights_files)

def intercept_DummyModelLoader_load_model(
    self, *, model_config: ModelConfig,
    device_config: DeviceConfig,
    lora_config: Optional[LoRAConfig],
    vision_language_config: Optional[VisionLanguageConfig],
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig) -> nn.Module:
    with set_default_torch_dtype(model_config.dtype):
        # with torch.device(device_config.device):
        model = _initialize_model(model_config, self.load_config,
                                    lora_config, vision_language_config).cuda()
        # For accurate performance evaluation, we assign
        # random values to the weights.
        initialize_dummy_weights(model)
        # cast quant matmul scales to float32
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)
            # FIXME: Remove this after Mixtral is updated
            # to use quant_method.
            if hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading()
    return model.eval()


InterceptObject.add_intercept(DefaultModelLoader,
                                  DefaultModelLoader._get_weights_iterator,
                                  intercept_get_weights_iterator)
InterceptObject.add_intercept(DummyModelLoader,
                                  DummyModelLoader.load_model,
                                  intercept_DummyModelLoader_load_model)