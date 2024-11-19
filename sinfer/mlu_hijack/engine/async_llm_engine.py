

from sinfer.engine.async_llm_engine import AsyncLLMEngine
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.logger import init_logger

logger = init_logger(__name__)


# for client pulling server scheduler profile data
def sinfer__engine__async_llm_engine__AsyncLLMEngine__save_scheduler_view(self):
    if hasattr(self.engine.scheduler, "__del__"):
        self.engine.scheduler.__del__()


MluHijackObject.add_hijack_object(AsyncLLMEngine,
                                  "save_scheduler_view",
                                  sinfer__engine__async_llm_engine__AsyncLLMEngine__save_scheduler_view)