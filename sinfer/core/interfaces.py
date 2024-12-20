import enum
from abc import ABC, abstractmethod
from typing import Dict, List
from typing import Sequence as GenericSequence
from sinfer.request import Request, Task

class AllocStatus(enum.Enum):
    """BlockSpaceManager.can_allocate的结果

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class SpaceManager(ABC):
    @staticmethod
    def get_block_space_manager():
        from sinfer.core.block_manager import _BlockSpaceManager
        return _BlockSpaceManager

    @abstractmethod
    def can_allocate(self, seq_group: Task) -> AllocStatus:
        pass

    @abstractmethod
    def allocate(self, seq_group: Task) -> None:
        pass

    @abstractmethod
    def can_append_slots(self, seq_group: Task,
                         num_lookahead_slots: int) -> bool:
        pass

    @abstractmethod
    def append_slots(
        self,
        seq: Request,
        num_lookahead_slots: int,
    ) -> Dict[int, List[int]]:
        pass

    @abstractmethod
    def fork(self, parent_seq: Request, child_seq: Request) -> None:
        pass

    @abstractmethod
    def can_swap_in(self, seq_group: Task,
                    num_lookahead_slots: int) -> AllocStatus:
        pass

    @abstractmethod
    def swap_in(self, seq_group: Task,
                num_lookahead_slots: int) -> Dict[int, int]:
        pass

    @abstractmethod
    def can_swap_out(self, seq_group: Task) -> bool:
        pass

    @abstractmethod
    def swap_out(self, seq_group: Task) -> Dict[int, int]:
        pass

    @abstractmethod
    def free(self, seq: Request) -> None:
        pass

    @abstractmethod
    def get_block_table(self, seq: Request) -> List[int]:
        pass

    @abstractmethod
    def get_num_free_gpu_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_cpu_blocks(self) -> int:
        pass

    @abstractmethod
    def access_all_blocks_in_seq(
        self,
        seq: Request,
        access_time: float,
    ) -> None:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, seqs: List[Request]) -> GenericSequence[int]:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, seq_group: Task):
        pass
