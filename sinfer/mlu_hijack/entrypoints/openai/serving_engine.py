
from http import HTTPStatus
from typing import Optional, Union

from sinfer.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest, ErrorResponse)
from sinfer.entrypoints.openai.serving_engine import OpenAIServing
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.logger import init_logger

logger = init_logger(__name__)


async def sinfer__entrypoints__openai__serving_engine__OpenAIServing___check_model(
    self, request: Union[CompletionRequest, ChatCompletionRequest]
) -> Optional[ErrorResponse]:
    if request.model in self.served_model_names:
        return None
    if request.model in [lora.lora_name for lora in self.lora_requests]:
        return None
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: when client send a request with model=save_scheduler_view,
            scheduler will dump profile data.
    '''
    if request.model == "save_scheduler_view":
        self.engine.save_scheduler_view()
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return self.create_error_response(
        message=f"The model `{request.model}` does not exist.",
        err_type="NotFoundError",
        status_code=HTTPStatus.NOT_FOUND)


MluHijackObject.add_hijack_object(OpenAIServing,
                                  OpenAIServing._check_model,
                                  sinfer__entrypoints__openai__serving_engine__OpenAIServing___check_model)