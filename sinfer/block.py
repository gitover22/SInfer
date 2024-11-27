from typing import List
from sinfer.utils import Device

_BLANK_TOKEN_ID = -1
DEFAULT_LAST_ACCESSED_TIME = -1

class LogicalTokenBlock:
    """
    一个KV cache的逻辑块
    """

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.block_number = block_number # 逻辑块编号
        self.block_size = block_size # 逻辑块的大小（slot数量, 默认16）
 
        self.token_ids = [_BLANK_TOKEN_ID] * block_size # 以列表形式存储具体的 token 数据, 未使用的部分用 _BLANK_TOKEN_ID（即 -1）填充。
        self.num_tokens = 0 # 当前逻辑块中已存储的 token 数量

    def is_empty(self) -> bool:
        """判断当前逻辑块是为空"""
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        """当前逻辑块的空余槽位"""
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        """判断当前逻辑块是否已经被装满"""
        return self.num_tokens == self.block_size
        
    def append_tokens(self, token_ids: List[int]) -> None:
        """将给定的一些token_ids装入当前逻辑块中"""
        assert len(token_ids) <= self.get_num_empty_slots() # 给定的token_ids的长度必须 <= 当前逻辑块剩余的槽位
        curr_idx = self.num_tokens  # 当前逻辑块第一个空槽的序号s
        self.token_ids[curr_idx:curr_idx + len(token_ids)] = token_ids  # 将tokens装到逻辑块的空槽中
        self.num_tokens += len(token_ids)  # 更新当前逻辑块中tokens的数量

    def get_token_ids(self) -> List[int]:
        """获取当前逻辑块中所有被装满的位置的token_ids"""
        return self.token_ids[:self.num_tokens]

    def get_last_token_id(self) -> int:
        """获取当前逻辑块所有被装满的位置的最后一个token_id"""
        assert self.num_tokens > 0
        return self.token_ids[self.num_tokens - 1]

class PhysicalTokenBlock:
    """
    一个物理块
    """

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        self.device = device  # 该物理块所在的设备
        self.block_number = block_number # 该物理块在对应设备上的全局block index
        self.block_size = block_size # 块的尺寸（即槽位数量，默认为16）
        self.block_hash = block_hash # 此块的哈希值（在prefix caching场景下使用，非此场景则附值为-1）
        self.num_hashed_tokens = num_hashed_tokens # 该物理块的hash值是由多少个前置token计算而来的（prefix caching场景下使用，非此场景则附值为0）

        self.ref_count = 0 # 引用计数：被多少个逻辑块引用
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME  # 记录此块最后一次被访问的时间，prefix caching场景下使用

        self.computed = False # 该物理块是否被计算过（prefix caching场景下使用）

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed})')

# 块表
BlockTable = List[PhysicalTokenBlock]
