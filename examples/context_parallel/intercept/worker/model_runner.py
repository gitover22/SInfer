
import torch
from typing import List, Optional
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.Intercept.mlu_definitions import check_latency_debug_with_device
from sinfer.worker.model_runner import (ModelRunner, _PAD_SLOT_ID)
from sinfer.request import TaskMetadata, SamplerOutput
from ..zigzag_utils import get_context_model_parallel_world_size, zigzag_split
import sinfer.envs as envs


@torch.inference_mode()
def sinfer__worker__model_runner__ModelRunner__execute_model(
    self,
    seq_group_metadata_list: List[TaskMetadata],
    kv_caches: List[torch.Tensor],
) -> Optional[SamplerOutput]:
    (input_tokens, input_positions, attn_metadata, sampling_metadata,
        lora_requests, lora_mapping, multi_modal_input
        ) = self.prepare_input_tensors(seq_group_metadata_list)

    if self.lora_config:
        self.set_active_loras(lora_requests, lora_mapping)

    # Currently cuda graph is only supported by the decode phase.
    prefill_meta = attn_metadata.prefill_metadata
    decode_meta = attn_metadata.decode_metadata
    if prefill_meta is None and decode_meta.use_cuda_graph:
        graph_batch_size = input_tokens.shape[0]
        model_executable = self.graph_runners[graph_batch_size]
    else:
        model_executable = self.model
    execute_model_kwargs = {
        "input_ids": input_tokens,
        "positions": input_positions,
        "kv_caches": kv_caches,
        "attn_metadata": attn_metadata,
    }
    if self.vision_language_config:
        execute_model_kwargs.update({"image_input": multi_modal_input})
    ''' 
    =============================
    Added by sinfer_mlu
    =============================
    @brief: insert timestamp markers
    '''
    if check_latency_debug_with_device():
        use_cuda_graph = (prefill_meta is None and decode_meta.use_cuda_graph)
        # if use_cuda_graph, the start timestamp will be inserted inside CUDAGraphRunner.forward()
        if not use_cuda_graph:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
    '''
    ==================
    End of addition
    ==================
    '''
    
    
    '''
    ==========================
    Modify by Context Parallel
    ==========================
    @brief: context parallel split input for model with zigzag method
    '''
    if get_context_model_parallel_world_size() > 1 and attn_metadata.prefill_metadata:
        zigzag_input_ids, zigzag_positions, zigzag_attn_metadata = \
            zigzag_split(execute_model_kwargs["input_ids"],
                            execute_model_kwargs["positions"],
                            execute_model_kwargs["attn_metadata"],
                            _PAD_SLOT_ID)
        execute_model_kwargs.update({"input_ids": zigzag_input_ids})
        execute_model_kwargs.update({"positions": zigzag_positions})
        execute_model_kwargs.update({"attn_metadata": zigzag_attn_metadata})
    '''
    =======================
    End of Context Parallel
    =======================
    '''
        
    hidden_states = model_executable(**execute_model_kwargs)
    
    
    #################################################################################################
    #                                           DEBUG                                               #
    #################################################################################################
    # import os
    # from sinfer.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
    # from ..distributed.parallel_state import get_context_model_parallel_rank
    # from ..zigzag_utils import context_parallel_tensor_all_gather, diff1
    # if get_context_model_parallel_world_size() > 1 and attn_metadata.prefill_metadata:
    #     hidden_states = context_parallel_tensor_all_gather(hidden_states, zigzag_attn_metadata, dim=0)
    # if attn_metadata.prefill_metadata and (kv_caches[0] is not None):
    #     file_path = '/workspace/output_base_' + str(hidden_states.shape) + \
    #                 '_tp_' + str(get_tensor_model_parallel_world_size()) + '.pth'
    #     if get_context_model_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
    #         if os.path.exists(file_path):
    #             print("##################compare################")
    #             hidden_states_base = torch.load(file_path)
    #             print("########output_diff1: ", diff1(hidden_states, hidden_states_base))
    #         else:
    #             print("##################save base################")
    #             torch.save(hidden_states, file_path)
                
    
    '''
    ==========================
    Modify by Context Parallel
    ==========================
    @brief: logits_processor in context parallel need attn_metadata param
    '''            
    if get_context_model_parallel_world_size() > 1 and attn_metadata.prefill_metadata:
        setattr(self.model.logits_processor, 'attn_metadata', zigzag_attn_metadata)
    else:
        setattr(self.model.logits_processor, 'attn_metadata', None)
    '''
    =======================
    End of Context Parallel
    =======================
    '''
    

    # Compute the logits.
    logits = self.model.compute_logits(hidden_states, sampling_metadata)

    ''' 
    =============================
    Added by sinfer_mlu
    =============================
    @brief: introduce a new instance variable time_markers and set it
    '''
    if check_latency_debug_with_device():
        end_marker = torch.cuda.Event(enable_timing=True)
        end_marker.record()
        if use_cuda_graph:
            self.time_markers = (model_executable.start, end_marker)
        else:
            self.time_markers = (start, end_marker)
    '''
    ==================
    End of addition
    ==================
    '''

    # Only perform sampling in the driver worker.
    if not self.is_driver_worker:
        return None

    # Sample the next token.
    output = self.model.sample(
        logits=logits,
        sampling_metadata=sampling_metadata,
    )

    return output


InterceptObject.add_intercept(ModelRunner,
                                  ModelRunner.execute_model,
                                  sinfer__worker__model_runner__ModelRunner__execute_model)