from typing import Optional
import torch
import sinfer
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from sinfer.distributed import broadcast
from sinfer.model_executor.sampling_metadata import SamplingMetadata
from sinfer.attention import AttentionMetadata
from sinfer.model_executor.layers.logits_processor import LogitsProcessor, _prune_hidden_states, _apply_logits_processors
from ...distributed.parallel_state import get_context_model_parallel_world_size, get_context_model_parallel_rank, get_tensor_model_parallel_world_size


def sinfer__module_executor__layers__logits_processor__LogitsProcessor__forward_wraper(
    self,
    embedding: torch.Tensor,
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    embedding_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if self.logits_as_input:
        logits = hidden_states
    else:
        '''
        ==========================
        Modify by Context Parallel
        ==========================
        @brief: context parallel requires special handling of hidden_states and logits
        '''
        if self.attn_metadata and get_context_model_parallel_world_size() > 1:
            hidden_states = _prune_hidden_states_context_parallel(hidden_states, sampling_metadata, self.attn_metadata)
        else:
            hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)
        
        # if cp rank > 0, skip self._get_logits, otherwise an error will be reported
        if get_context_model_parallel_world_size() > 1 and \
            get_context_model_parallel_rank() > 0:
            logits = None
            return logits
        '''
        =======================
        End of Context Parallel
        =======================
        '''

        # Get the logits for the next tokens.
        logits = self._get_logits(hidden_states, embedding, embedding_bias)

    if logits is not None:
        logits *= self.scale

        # Apply logits processors (if any).
        logits = _apply_logits_processors(logits, sampling_metadata)

    return logits

'''
==========================
Modify by Context Parallel
==========================
@brief: token num can be divisible by context_parallel_size * 2 after padding,
        and then split to context parallel groups with zigzag method, now we
        need to find the last valid tokens, and get the logits for the next tokens.
'''
def _prune_hidden_states_context_parallel(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    attn_metadata: AttentionMetadata
) -> torch.Tensor:
    select_hidden_states_list = []
    seq_start_loc = attn_metadata.prefill_metadata.seq_start_loc
    batch_num = seq_start_loc.shape[0] - 1
    for batch in range(batch_num):
        start = seq_start_loc[batch]
        end = seq_start_loc[batch + 1]
        hidden_states_ = hidden_states[start : end]
        split_seq_len = hidden_states_.shape[0] // 2
        seq_len = attn_metadata.prefill_metadata.seq_lens[batch]
        last_id = seq_len - 1
        idx = last_id // split_seq_len
        select_hidden_states = torch.zeros((1, hidden_states.shape[-1]), dtype = hidden_states.dtype, device = hidden_states.device)
        if idx < get_context_model_parallel_world_size():
            target_cp_id = idx
            src_rank = get_tensor_model_parallel_world_size() * target_cp_id
            if get_context_model_parallel_rank() == target_cp_id:
                selected_token_indices = last_id - idx * split_seq_len
                select_hidden_states = hidden_states_[selected_token_indices].unsqueeze(0)
        else:
            target_cp_id = get_context_model_parallel_world_size() * 2 - 1 - idx
            src_rank = get_tensor_model_parallel_world_size() * target_cp_id
            if get_context_model_parallel_rank() == target_cp_id:
                selected_token_indices = last_id - idx * split_seq_len + split_seq_len
                select_hidden_states = hidden_states_[selected_token_indices].unsqueeze(0)

        select_hidden_states = broadcast(select_hidden_states, src = src_rank)
        select_hidden_states_list.append(select_hidden_states)
        
    select_hidden_states = torch.cat(select_hidden_states_list, dim=0)
    return select_hidden_states
'''
=======================
End of Context Parallel
=======================
'''
    

MluHijackObject.add_hijack_object(LogitsProcessor,
                                  LogitsProcessor.forward,
                                  sinfer__module_executor__layers__logits_processor__LogitsProcessor__forward_wraper)