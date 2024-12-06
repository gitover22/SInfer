from sinfer.sequence import SequenceGroup, SequenceStatus
from sinfer.Intercept.mlu_definitions import USE_PAGED
from sinfer.Intercept.mlu_hijack_utils import MluHijackObject
from sinfer.core.block_manager import _BlockSpaceManager
from sinfer.logger import init_logger

logger = init_logger(__name__)


def sinfer__core__block_manager_v1__BlockSpaceManagerV1__can_append_slots(self,
                      seq_group: SequenceGroup,
                      num_lookahead_slots: int = 0) -> bool:
    assert (num_lookahead_slots == 0
            ), "lookahead allocation not supported in BlockSpaceManagerV1"
    # 优化分配策略: unpagged mode 可以返回true
    if not USE_PAGED:
        return True
    else:
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

MluHijackObject.add_hijack_object(_BlockSpaceManager,
                                  _BlockSpaceManager.can_append_slots,
                                  sinfer__core__block_manager_v1__BlockSpaceManagerV1__can_append_slots)