from typing import Optional, Tuple, Union, Type
import torch
import sinfer
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from bangtransformer.torch import bt_ops
from sinfer.model_executor.layers.rotary_embedding import RotaryEmbedding
from ...distributed.parallel_state import get_context_model_parallel_world_size

def sinfer__module_executor__layers__rotary_embedding__RotaryEmbedding__forward_wraper(
    self,
    positions: torch.Tensor,
    x: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
    # ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: use bt_ops.apply_rotary to replace ops.rotary_embedding
    '''
    if offsets is not None:
        raise ValueError(f"bt_ops.apply_rotary not support offsets yet.")
    else:
        if RotaryEmbedding.set_cos_sin == False:
            RotaryEmbedding.cos_, RotaryEmbedding.sin_ = self._get_cos_sin(positions)
            RotaryEmbedding.set_cos_sin = True
        interleaved = True
        if self.is_neox_style:
            interleaved = False
        '''
        ==========================
        Modify by Context Parallel
        ==========================
        @brief: context parallel need discrete = True
        '''
        position_ids = None if (RotaryEmbedding.is_prompt and get_context_model_parallel_world_size == 1) else positions
        discrete = False if (RotaryEmbedding.is_prompt and get_context_model_parallel_world_size == 1) else True
        '''
        =======================
        End of Context Parallel
        =======================
        '''
        x = bt_ops.apply_rotary(x,
                                RotaryEmbedding.sin_,
                                RotaryEmbedding.cos_,
                                position_ids,
                                RotaryEmbedding.cu_seq_len,
                                interleaved,
                                discrete,
                                False,
                                RotaryEmbedding.max_seq_len)
    return x
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


MluHijackObject.add_hijack_object(RotaryEmbedding,
                                  RotaryEmbedding.forward,
                                  sinfer__module_executor__layers__rotary_embedding__RotaryEmbedding__forward_wraper)