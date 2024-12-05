import sinfer

from torch_mlu.utils.model_transfer import transfer
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from sinfer.Intercept.mlu_hijack import SinferMluHijack
     
MluHijackObject.undo_hijack()

print("Apply Context Parallel Demo!")
from . import distributed
from . import attention
from . import config
from . import engine
from . import entrypoints
from . import model_executor
from . import worker

MluHijackObject.apply_hijack()