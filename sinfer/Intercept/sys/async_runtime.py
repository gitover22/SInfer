

from sinfer.sys.async_runtime import AsyncLLMRuntime
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.logger import init_logger

logger = init_logger(__name__)


# for client pulling server scheduler profile data
def intercept_AsyncLLMEngine_save_scheduler_view(self):
    if hasattr(self.engine.scheduler, "__del__"):
        self.engine.scheduler.__del__()


InterceptObject.add_intercept(AsyncLLMRuntime,
                                  "save_scheduler_view",
                                  intercept_AsyncLLMEngine_save_scheduler_view)