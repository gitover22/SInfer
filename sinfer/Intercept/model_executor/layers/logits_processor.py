
from typing import Optional

import torch
from sinfer.logger import init_logger
from sinfer.model_executor.layers.logits_processor import LogitsProcessor
from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from sinfer.Intercept.distributed.communication_op import intercept_tensor_model_parallel_gather


logger = init_logger(__name__)


def intercept_get_logits(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    '''
    @brief: use custom gather to reduce host time for non driver worker.
    NOTE: this hijack function should be REMOVED when torch upgrade to 2.4.0
    '''
    logits = intercept_tensor_model_parallel_gather(logits)

    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[:, :self.org_vocab_size]
    return logits


InterceptObject.add_intercept(LogitsProcessor,
                                  LogitsProcessor._get_logits,
                                  intercept_get_logits)
