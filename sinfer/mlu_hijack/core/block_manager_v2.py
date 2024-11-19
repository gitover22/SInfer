from sinfer.sequence import SequenceGroup, SequenceStatus
from sinfer.mlu_hijack.mlu_definitions import USE_PAGED
from sinfer.mlu_hijack.mlu_hijack_utils import MluHijackObject
from sinfer.core.block_manager_v2 import BlockSpaceManagerV2
from sinfer.utils import Device
from sinfer.logger import init_logger

logger = init_logger(__name__)


def sinfer__core__block_manager_v2__BlockSpaceManagerV2__can_append_slots(self, seq_group: SequenceGroup,
                        num_lookahead_slots: int) -> bool:
    """Determine if there is enough space in the GPU KV cache to continue
    generation of the specified sequence group.

    We use a worst-case heuristic: assume each touched block will require a
    new allocation (either via CoW or new block). We can append slots if the
    number of touched blocks is less than the number of free blocks.

    "Lookahead slots" are slots that are allocated in addition to the slots
    for known tokens. The contents of the lookahead slots are not defined.
    This is used by speculative decoding when speculating future tokens.
    """
    '''
    =============================
    Modify by sinfer_mlu
    =============================
    @brief: optimize the allocation strategy for unpagged mode
    '''
    if not USE_PAGED:
        return True
    else:
        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


MluHijackObject.add_hijack_object(BlockSpaceManagerV2,
                                  BlockSpaceManagerV2.can_append_slots,
                                  sinfer__core__block_manager_v2__BlockSpaceManagerV2__can_append_slots)