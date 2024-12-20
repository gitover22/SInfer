import os
import sinfer

from torch_mlu.utils.model_transfer import transfer
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.logger import init_logger
from sinfer.model_executor.models import ModelRegistry
from sinfer.Intercept.model_executor.custom_model.custom import CustomForCausalLM
from transformers import AutoConfig
from sinfer.Intercept.transformers_utils.configs import CustomConfig

logger = init_logger(__name__)


class SinferMluIntercept:
    def __init__(self):
        pass

    def apply_optimizations(self, paged=False):
        logger.info("intercepting Layer opend!")
        import sinfer.Intercept.attention
        import sinfer.Intercept.distributed
        import sinfer.Intercept.model_executor
        import sinfer.Intercept.worker
        InterceptObject.apply_intercept()

    def undo_optimization(self):
        logger.info("intercepting Layer closed!")
        InterceptObject.remove_intercept()

sinfer_mlu_intercept = SinferMluIntercept()
sinfer_mlu_intercept.apply_optimizations(USE_PAGED)


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
    importlib.import_module("examples.context_parallel.intercept.mlu_hijack") # 动态导入模块

if check_context_parallel():
    use_context_parallel_hijack()
