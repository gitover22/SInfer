import sinfer

from torch_mlu.utils.model_transfer import transfer
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.Intercept.mlu_intercept import SinferMluIntercept
     
InterceptObject.remove_intercept()

print("Apply Context Parallel Demo!")
from . import distributed
from . import attention
from . import config
from . import engine
from . import entrypoints
from . import model_executor
from . import worker

InterceptObject.apply_intercept()