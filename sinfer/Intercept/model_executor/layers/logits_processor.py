
from typing import Optional

import torch
from sinfer.logger import init_logger
from sinfer.model_executor.layers.logits_processor import LogitsProcessor
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from sinfer.Intercept.distributed.communication_op import sinfer__distributed__communication_op__tensor_model_parallel_gather


logger = init_logger(__name__)


def sinfer__model_executor__layers__logits_processor__LogitsProcessor___get_logits(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: use custom gather to reduce host time for non driver worker.
    NOTE: this hijack function should be REMOVED when torch upgrade to 2.4.0
    '''
    logits = sinfer__distributed__communication_op__tensor_model_parallel_gather(logits)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[:, :self.org_vocab_size]
    return logits


MluHijackObject.add_hijack_object(LogitsProcessor,
                                  LogitsProcessor._get_logits,
                                  sinfer__model_executor__layers__logits_processor__LogitsProcessor___get_logits)
