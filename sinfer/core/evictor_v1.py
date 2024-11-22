import enum
from abc import ABC, abstractmethod
from typing import OrderedDict

from sinfer.block import PhysicalTokenBlock

class EvictionPolicy(enum.Enum):
    """
    驱逐策略的枚举类
    """
    LRU = enum.auto()


class Evictor(ABC):
    """
    BlockAllocator类应该使用Evictor子类来处理释放的物理token块的回收
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> PhysicalTokenBlock:
        """Runs the eviction algorithm and returns the evicted block"""
        pass

    @abstractmethod
    def add(self, block: PhysicalTokenBlock):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        """Simply removes the block with the hash value block_hash from the
        evictor. Caller is responsible for making sure that block_hash is
        contained in the evictor before calling remove. Should be used to
        "bring back" blocks that have been freed but not evicted yet.
        """
        pass


class LRUEvictor(Evictor):
    """
    按照最近最少使用的顺序进行淘汰，依据记录在 `PhysicalTokenBlock` 中的 `last_accessed` 时间戳。
    如果存在多个块的 `last_accessed` 时间相同，则淘汰 `num_hashed_tokens` 值最大的那个块。
    如果两个块同时具有最小的 `last_accessed` 时间和最大的 `num_hashed_tokens` 值，则会随机选择一个进行淘汰。
    """

    def __init__(self):
        # mapping :{block_hash:block}
        self.free_table: OrderedDict[int, PhysicalTokenBlock] = OrderedDict() # 用于存放暂时不用的物理块

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        '''
        从free_table中驱逐一个物理块
        '''
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_block = next(iter(self.free_table.values()))
        # 具有最小时间戳的块应连续放置在 OrderedDict 的开头。遍历所有这些块，找到哈希令牌数量最多的那个块。
        for _, block in self.free_table.items():
            if evicted_block.last_accessed < block.last_accessed: # 可能最后一次使用时间相同
                break
            if evicted_block.num_hashed_tokens < block.num_hashed_tokens:
                evicted_block = block

        self.free_table.pop(evicted_block.block_hash) # 从驱逐器的free_table中移除被选中的驱逐物理块

        evicted_block.computed = False # 将被驱逐的物理块的计算状态设为False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        """
        将拥有指定hash值的物理块从free_tables中移除
        """
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table[block_hash]
        self.free_table.pop(block_hash)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
