import torch

from sinfer.distributed import communication_op
from sinfer.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tensor_model_parallel_rank,
                                             get_tensor_model_parallel_world_size)
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject


def sinfer__distributed__communication_op__tensor_model_parallel_gather(
        input_: torch.Tensor,
        dst: int = 0,
        dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    '''
    @brief: use p2p communication to reduce gather host time for non driver worker.
    NOTE: this hijack function should be REMOVED when torch upgrade to 2.4.0
    '''
    rank = get_tensor_model_parallel_rank()

    gather_list = None
    if rank == dst:
        gather_list = [
            torch.empty_like(input_) for _ in range(dst)
        ] + [input_] + [
            torch.empty_like(input_) for _ in range(dst + 1, world_size)
        ]

    send_recv_op_list = []
    if rank != dst:
        op = torch.distributed.P2POp(torch.distributed.isend,
                                     input_,
                                     dst,
                                     group=get_tensor_model_parallel_group())
        send_recv_op_list.append(op)
    else:
        for r in range(0, world_size):
            if r == dst:
                continue
            op = torch.distributed.P2POp(torch.distributed.irecv,
                                         gather_list[r],
                                         r,
                                         group=get_tensor_model_parallel_group())
            send_recv_op_list.append(op)
    reqs = torch.distributed.batch_isend_irecv(send_recv_op_list)
    for req in reqs:
        req.wait()

    if get_tensor_model_parallel_rank() == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor


MluHijackObject.add_hijack_object(communication_op,
                                  communication_op.tensor_model_parallel_gather,
                                  sinfer__distributed__communication_op__tensor_model_parallel_gather)
