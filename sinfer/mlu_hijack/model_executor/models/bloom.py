from typing import Optional

import torch
from torch import nn
from transformers import BloomConfig

from sinfer.logger import init_logger
from sinfer.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from sinfer.mlu_hijack.model_executor.layers.feed_forward import FeedForward
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject

from sinfer.model_executor.models.bloom import BloomAttention, BloomBlock

logger = init_logger(__name__)


def sinfer__module_executor__models__baichuan__BloomBlock__init__(
        self,
        config: BloomConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
    super(BloomBlock, self).__init__()
    hidden_size = config.hidden_size

    self.input_layernorm = nn.LayerNorm(hidden_size,
                                            eps=config.layer_norm_epsilon)
    self.self_attention = BloomAttention(config, quant_config)
    self.post_attention_layernorm = nn.LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon)
    # use FeedForward instead of MLP
    self.mlp = FeedForward(hidden_size=hidden_size,
                           intermediate_size=hidden_size * 4,
                           hidden_act='gelu',
                           up_proj_name="dense_h_to_4h",
                           is_gated=False,
                           down_proj_name="dense_4h_to_h",
                           bias=False,
                           quant_config=quant_config)
    
    self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)
    
MluHijackObject.add_hijack_object(BloomBlock,
                                  BloomBlock.__init__,
                                  sinfer__module_executor__models__baichuan__BloomBlock__init__)