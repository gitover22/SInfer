from typing import Dict, Optional, Sequence, List
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from .distributed.parallel_state import (
    get_context_model_parallel_rank, get_context_model_parallel_world_size, get_context_model_parallel_group)
from sinfer.distributed.utils import divide
from sinfer.attention import AttentionMetadata
import copy


def diff1(result: torch.Tensor, baseline: torch.Tensor):
        result = result.flatten().float().to('cpu')
        baseline = baseline.flatten().float().to('cpu')
        assert result.shape == baseline.shape
        error = torch.abs(baseline - result)
        denominator = torch.sum(torch.abs(baseline)).item()
        eps = 0.0 if denominator > 0 else 1e-9
        diff1 = torch.sum(error) / (denominator + eps)
        return diff1.item()


def get_pad_seq(seq_len: int, pad: int):
    return (seq_len // pad + (int)((seq_len) % (pad) > 0)) * pad


# Gather the partial results of a batch on context parallel groups
# together and place them in the order before zigzag splitting
def context_parallel_tensor_all_gather_(input_, dim=-1):
    world_size = get_context_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    
    assert input_size[dim] % 2 == 0, (f"input tensor split dim % 2 != 0")
    
    gather_list = [torch.empty(input_.shape, dtype=input_.dtype, device=input_.device) for _ in range(world_size)]
    torch.distributed.all_gather(
         gather_list, input_, group=get_context_model_parallel_group())
    
    first = []
    second = []
    for i in range(world_size):
        first_second = torch.split(gather_list[i], gather_list[i].shape[dim] // 2, dim=dim)
        first.append(first_second[0])
        second.insert(0, first_second[1])       
    tensor_list = first + second
    output_tensor = torch.cat(tensor_list, dim = dim).contiguous()    
    return output_tensor


# Gather the partial results of each batch on the context parallel groups together,
# place them in the order before zigzag splitting, and remove the pad part.
# This function is used for debugging
def context_parallel_tensor_all_gather(input, attn_metadata, dim=-1):
    if dim < 0:
        dim += input.dim()
    slice_ = ()
    for i in range(dim):
        slice_ + (slice(None))        
    select_list = []
    seq_start_loc = attn_metadata.prefill_metadata.seq_start_loc
    batch_num = seq_start_loc.shape[0] - 1
    for batch in range(batch_num):
        start = seq_start_loc[batch].item()
        end = seq_start_loc[batch + 1].item()
        slice1 = slice_ + (slice(start, end), )
        input_ = input[slice1]
        gather_ = context_parallel_tensor_all_gather_(input_, dim=dim)
        slice2 = slice_ + (slice(None, attn_metadata.prefill_metadata.seq_lens[batch]), )
        select = gather_[slice2]
        select_list.append(select)
    output = torch.cat(select_list, dim=dim)
    return output


# Pad one dimension of a tensor so that it is divisible by context_parallel_size * 2,
# and then use zigzag method to split it into different context parallel groups
def zigzag_split_(tensor: torch.Tensor, dim = -1, pad_value=0):
    if dim < 0:
        dim = tensor.dim() + dim
    split_num = get_context_model_parallel_world_size() * 2
    pad_num = get_pad_seq(tensor.shape[dim], split_num) - tensor.shape[dim]
    pad_param = (0, 0) * (tensor.dim() - dim - 1) +  (0, pad_num) + (0, 0) * dim
    tensor_pad = F.pad(tensor, pad_param, value = pad_value)
    split_size = divide(tensor_pad.size()[dim], split_num)
    # Split.
    tensor_list = torch.split(tensor_pad, split_size, dim = dim)
    first = tensor_list[get_context_model_parallel_rank()]
    second = tensor_list[split_num - get_context_model_parallel_rank() - 1]
    output_tensor  = torch.cat((first, second), dim=dim).contiguous()
    return output_tensor


# Split each batch of input_ids, positions, attn_metadata.slot_mapping with zigzag method,
# and update prefill_metadata.seq_start_loc and prefill_metadata.max_seq_len
def zigzag_split(input_ids: torch.Tensor,
                 positions: torch.Tensor,
                 attn_metadata: AttentionMetadata,
                 pad_slot_id: int):
    zigzag_input_ids: List[int] = []
    zigzag_positions: List[int] = []
    zigzag_slot_mapping: List[int] = []
    zigzag_attn_metadata = copy.deepcopy(attn_metadata)
    seq_lens: List[int] = []
    seq_start_loc = attn_metadata.prefill_metadata.seq_start_loc
    batch_num = seq_start_loc.shape[0] - 1
    for batch in range(batch_num):
        start, end = seq_start_loc[batch], seq_start_loc[batch + 1]
        input_ids_ = input_ids[start : end]
        positions_ = positions[start : end]
        zigzag_input_ids_ = zigzag_split_(input_ids_)
        zigzag_positions_ = zigzag_split_(positions_)
        zigzag_input_ids.append(zigzag_input_ids_)
        zigzag_positions.append(zigzag_positions_)
        seq_lens.append(zigzag_input_ids_.shape[0])
        slot_mapping_ = attn_metadata.slot_mapping[start : end]
        zigzag_slot_mapping_ = zigzag_split_(slot_mapping_, pad_value=pad_slot_id)
        zigzag_slot_mapping.append(zigzag_slot_mapping_)
        
    zigzag_input_ids = torch.cat(zigzag_input_ids, dim=0)
    zigzag_positions = torch.cat(zigzag_positions, dim=0)
    zigzag_slot_mapping = torch.cat(zigzag_slot_mapping, dim=0)
       
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens,
                                   dtype=torch.int,
                                   device=input_ids.device)
    seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device=input_ids.device)
    torch.cumsum(seq_lens_tensor,
                 dim=0,
                 dtype=seq_start_loc.dtype,
                 out=seq_start_loc[1:])
    
    zigzag_attn_metadata.prefill_metadata.seq_start_loc = seq_start_loc
    zigzag_attn_metadata.prefill_metadata.subquery_start_loc = seq_start_loc
    zigzag_attn_metadata.prefill_metadata.max_seq_len = max_seq_len
    zigzag_attn_metadata.slot_mapping = zigzag_slot_mapping
    
    return zigzag_input_ids, zigzag_positions, zigzag_attn_metadata
