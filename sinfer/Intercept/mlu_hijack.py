import os
import sinfer

from torch_mlu.utils.model_transfer import transfer
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from sinfer.logger import init_logger
from sinfer.model_executor.models import ModelRegistry
from sinfer.Intercept.model_executor.custom_model.custom import CustomForCausalLM
from transformers import AutoConfig
from sinfer.Intercept.transformers_utils.configs import CustomConfig

logger = init_logger(__name__)


class SinferMluHijack:
    def __init__(self):
        pass

    def apply_optimizations(self, paged=False):
        logger.info("Apply MLU optimization!")
        import sinfer.Intercept.attention
        import sinfer.Intercept.core
        if check_scheduler_profile_enable():
            import sinfer.Intercept.core.scheduler
            import sinfer.Intercept.sys.async_runtime
            import sinfer.Intercept.entrypoints.openai.serving_engine
        import sinfer.Intercept.distributed
        import sinfer.Intercept.sys
        import sinfer.Intercept.entrypoints
        import sinfer.Intercept.executor
        import sinfer.Intercept.model_executor
        import sinfer.Intercept.worker
        InterceptObject.apply_intercept()

    def undo_optimization(self):
        logger.info("Undo MLU optimization!")
        InterceptObject.remove_intercept()

sinfer_mlu_hijack = SinferMluHijack()
sinfer_mlu_hijack.apply_optimizations(USE_PAGED)


AutoConfig.register("custom", CustomConfig)
ModelRegistry.register_model("CustomForCausalLM", CustomForCausalLM)

def use_context_parallel_hijack():
    import importlib
    import os
    import sys
    from pathlib import Path
    file_path = Path(os.path.abspath(__file__))
    SInfer_dir = file_path.parent.parent.parent
    sys.path.append(str(SInfer_dir))
    importlib.import_module("examples.context_parallel.mlu_hijack.mlu_hijack") # 动态导入模块

if check_context_parallel():
    use_context_parallel_hijack()
