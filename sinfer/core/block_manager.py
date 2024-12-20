"""A block manager that manages token blocks."""
import math
from abc import ABC, abstractmethod
from itertools import count, takewhile
from os.path import commonprefix
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Set

from sinfer.block import BlockTable, PhysicalTokenBlock
from sinfer.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from sinfer.core.interfaces import AllocStatus, SpaceManager
from sinfer.logger import init_logger
from sinfer.request import Request, Task, RequestStatus
from sinfer.utils import Device

logger = init_logger(__name__)


class BlockAllocatorBase(ABC):
    @abstractmethod
    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        pass

    @abstractmethod
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def contains_block(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass


class CachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU) -> None:
        self.device = device
        self.block_size = block_size  # 每个block的大小
        self.num_blocks = num_blocks  # 该设备上留给KV cache的总blocks数量

        self.current_num_blocks = 0 # 当前设备上已分配的物理块的数量(如果一个物理块被释放，但refcount不为0，则current_num_blocks不会减少)
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {} # 用于存放正在被使用的物理块 {该物理块（也可以说是逻辑块）的hash值: 物理块}

        self.evictor: Evictor = make_evictor(eviction_policy)  # 默认是LRUEvictor

        self.default_hash_ctr = count()

    def allocate_block(self, block_hash: int,
                       num_hashed_tokens: int) -> PhysicalTokenBlock:
        """
        为哈希值为block_hash分配一个物理块
        Args:
            block_hash: 某个逻辑块的hash值。
                        对于一个seq，计算其下序号为logical_idx的逻辑块的hash值的方法如下：
                        hash(tuple(截止到序号为logical_idx为止这个seq的所有token_id，
                                   lora相关的配置)）
            num_hashed_tokens：对于一个seq，截止到序号为logical_idx的逻辑块为止
                             （包含这个逻辑块），一共有多少个token
        """

        # 如果目前已经使用的物理块数量 == 该设备上所有的物理块数量
        if self.current_num_blocks == self.num_blocks:
            block = self.evictor.evict() # 返回一个被选中的驱逐块block，这个block已经从驱逐器的free_tables中移走了
            # 设定当前block对应的block hash值，直接返回当前block
            block.block_hash = block_hash
            block.num_hashed_tokens = num_hashed_tokens
            return block
        # 如果目前已使用的物理块数量 ！= 该设备上所有的物理块数量，那么就新分配一个物理块
        block = PhysicalTokenBlock(device=self.device,
                                   block_number=self.current_num_blocks,
                                   block_size=self.block_size,
                                   block_hash=block_hash,
                                   num_hashed_tokens=num_hashed_tokens)
        self.current_num_blocks += 1 # 当前已使用的物理块数量 += 1
        return block

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        # 如果没有传入逻辑块hash值的话，就用一个伪hash值
        if block_hash is None:
            block_hash = next(self.default_hash_ctr)
        # 如果hash值在evictor的free_table中的话 说明拥有这个hash值的物理块，之前因为没人引用被释放了，就暂时存在free_tables中，
        # 现在来的这条seq拥有同样的prefix，所以这个物理块又可以被使用了
        if block_hash in self.evictor:
            assert block_hash not in self.cached_blocks # 首先要保证拥有这个hash值的物理块，肯定不在正在被使用的物理块字典中
            block = self.evictor.remove(block_hash) # 重新使用这个物理块了，因此我们将该物理块从free_tables中移除，返回的block就是该物理块
            assert block.ref_count == 0 # 必须保证此时该物理块没有引用任何逻辑块（因为你刚从驱逐器的冷宫中被释放出来）
            self.cached_blocks[block_hash] = block # 放进“正在被使用的物理块”字典中
            block.ref_count += 1
            assert block.block_hash == block_hash
            return block
        # 如果现在hash既不在驱逐器的free_tables中，也不在“当前正在被使用物理块“字典中的话
        # 说明该seq装的对应文本是之前没见过的，不满足prefix可以重复利用的场景
        if block_hash not in self.cached_blocks:
            # 为这个seq，分配一个新的物理块
            self.cached_blocks[block_hash] = self.allocate_block(
                block_hash, num_hashed_tokens)
        # 从正在使用的物理块字典（cached_blocks）中取出这个新分配的物理块，
        # 并将其ref_count += 1，最后返回它
        block = self.cached_blocks[block_hash]
        assert block.block_hash == block_hash
        block.ref_count += 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            assert block.block_hash not in self.evictor
            self.evictor.add(block)

            # Remove the block from the cached_blocks
            del self.cached_blocks[block.block_hash]

    def get_num_free_blocks(self) -> int:
        return (self.num_blocks - self.current_num_blocks +
                self.evictor.num_blocks)

    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def contains_block(self, block_hash: int) -> bool:
        return block_hash in self.cached_blocks or block_hash in self.evictor

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        # Update the hash of block and the cached_blocks dictionary.
        assert not self.contains_block(block_hash)
        old_hash = block.block_hash
        block.block_hash = block_hash
        del self.cached_blocks[old_hash]
        self.cached_blocks[block_hash] = block


class UncachedBlockAllocator(BlockAllocatorBase):
    """
        管理一个设备上的空闲物理块
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device  # cpu/gpu
        self.block_size = block_size # 物理块大小
        self.num_blocks = num_blocks # 该设备上留给KV cache的总物理块数量
        self.free_blocks: BlockTable = [] # 跟踪该设备上所有未被使用过的物理块

        # 初始化物理块列表free_blocks
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_blocks.append(block)

    # 从空闲块中取出一个物理块
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    # 释放一个物理块，会检查引用计数器
    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    # 获取空闲块的数量
    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    # 获取一个设备的物理块总数
    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")


class _BlockSpaceManager(SpaceManager):
    """管理逻辑块和物理块之间的映射"""

    def __init__(
        self,
        block_size: int,     # 每个block的槽位大小
        num_gpu_blocks: int, # 当前gpu上最多能分配的block数量
        num_cpu_blocks: int, # 当前cpu上，用于做swap的内存中，最多能分配的block数量
        watermark: float = 0.01, # 内存交换的水位线（阈值）
        sliding_window: Optional[int] = None, # 滑动窗口的大小
        enable_caching: bool = False, # 是否需要做prefix caching
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        if enable_caching and sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not allowed with prefix caching enabled!")

        self.block_sliding_window = None
        if sliding_window is not None:
            # Round up to nearest block size to regularize sliding window
            # allocation sizes.
            self.block_sliding_window = math.ceil(sliding_window / block_size)

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        # 水位线block数量：这个阈值决定是否要给当前seq分配block
        # 设置水位线block的目的是不要一下打满设备中的物理块，留一些buffer，避免后续频繁地发生swap
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # 根据是否开启了prefix caching，来选择不同的allocator
        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            self.gpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator = UncachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
            
        # seq_id -> BlockTable.维护每个seq下的物理块列表
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: Task) -> AllocStatus:
        '''
        是否可以给这个处于waiting队列的seq_group分配物理块做prefill, 返回结果有三种情况：
        - AllocStatus.NEVER 永不分配
        - AllocStatus.OK 可以分配
        - AllocStatus.LATER 延迟分配
        '''
        seq = seq_group.get_seqs(status=RequestStatus.WAITING)[0] # waiting队列中都是prefill阶段的序列，且只有一个序列
        num_required_blocks = len(seq.logical_token_blocks) # 这个序列的所有逻辑块数量

        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks() # 获取所有可用物理块数量

        # 设备中所有的物理块数量 - 该seq实际需要的物理块数量 < 水位线block数量，则不分配（说明当前seq太长了）
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:  # 如果设备中可用的物理块数量 - 该seq实际需要的block数量 >= 水位线block数量，则分配
            return AllocStatus.OK
        else:
            return AllocStatus.LATER         # 否则，现在不能分配，但可以延迟分配

    def allocate(self, seq_group: Task) -> None:
        """
        为当前seq_group分配物理块做prefill
        """
        seq = seq_group.get_seqs(status=RequestStatus.WAITING)[0] # waiting队列没做过prefill，只有一条序列

        # 计算该seq的逻辑块数量
        # （prefill阶段，有多少个逻辑块，就应该分配多少个物理块）
        num_prompt_blocks = len(seq.logical_token_blocks)

        block_table: BlockTable = []

        # 遍历该seq的所有逻辑块
        for logical_idx in range(num_prompt_blocks):
            # 如果block的滑动窗口长度不为空
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
                # Set the reference counts of the token blocks.
                block.ref_count = seq_group.num_seqs()
            # 如果做了prefix caching，即使用的是CachedBlockAllocator
            elif self.enable_caching:
                block = self.gpu_allocator.allocate(
                    seq.hash_of_block(logical_idx),
                    seq.num_hashed_tokens_of_block(logical_idx))
            # UncachedBlockAllocator对应的情况
            else:
                block = self.gpu_allocator.allocate()
                # Set the reference counts of the token blocks.
                block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # prefill阶段，这个seq_group下所有的seq共享一个prompt，也即共享这个prompt代表的物理块
        for seq in seq_group.get_seqs(status=RequestStatus.WAITING):
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slots(self,
                         seq_group: Task,
                         num_lookahead_slots: int = 0) -> bool:
        '''能否为seq_group分配物理块做decode
        检查对于SG中的每一个seq, 是否能至少分配一个空闲物理块给它(考虑最坏情况)
        '''
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerV1"
        
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=RequestStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def _promote_last_block(
        self,
        seq: Request,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        assert self.enable_caching

        # 如果当前最后一个逻辑块已经是满的，那么就根据所有tokens计算一个new_hash值
        new_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)

        # 如果根据new_hash，这最后一个物理块在cached_tables(有人正在和你用一样的)
        # 或者在驱逐器中
        if self.gpu_allocator.contains_block(new_hash):
            self.gpu_allocator.free(last_block) # 释放最后一个物理块
            return self.gpu_allocator.allocate(new_hash) # 用new_hash重新分配（即复用）物理块
        # 如果根据new_hash，这最后一个物理块既不在cached_tables也不在驱逐器中
        # （说明只有它的默认hash值在cached_tables中）
        else:
            self.gpu_allocator.update_hash(new_hash, last_block) # 对于这个物理块，在cached_tables中用new_hash代替old_hash
            return last_block

    def _is_last_block_full(
        self,
        seq: Request,
    ) -> bool:
        token_ids_len = seq.data.get_len()
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def _maybe_promote_last_block(
        self,
        seq: Request,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        # 检查当前这最后一个逻辑块是否满了，如果是，则检查是否有可复用物理块
        if self._is_last_block_full(seq):
            return self._promote_last_block(seq, last_block)
        else:
            return last_block

    def _allocate_last_physical_block(
        self,
        seq: Request,
    ) -> PhysicalTokenBlock:
        '''
        在想添加新的物理块之前，调用这个函数，来判断是不是真得有必要添加一个物理块
        '''
        if not self.enable_caching:
            return self.gpu_allocator.allocate() # 如果不使用prefix caching，就直接分配物理块
        block_hash: Optional[int] = None
        # 如果该seq的最后一块逻辑块是满的
        if (self._is_last_block_full(seq)):
            # 那么就截止到最后一块逻辑块（包括最后一块逻辑为止）计算所有token的block hash
            # (除了prefill阶段外，decode阶段只有在逻辑块满的时候才重新计算整串tokens的hash值，
            # 否则就用默认hash值)
            block_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)
        num_hashed_tokens = seq.num_hashed_tokens_of_block(
            len(seq.logical_token_blocks) - 1)

        # 使用prefix caching下的allocate方法，根据物理块的hash值，检查当前
        # 是否有可复用的物理块，以此决定是新分配一个物理块，还是复用旧物理块。
        new_block = self.gpu_allocator.allocate(block_hash, num_hashed_tokens)

        # If the block has is None, then the block is not full.
        # If the block is not full, then we expect it to have a refcount of 1.
        if block_hash is None:
            assert new_block.ref_count == 1
        return new_block

    def append_slots(
        self,
        seq: Request,
        num_lookahead_slots: int = 0,
    ) -> Dict[int, List[int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        # 如果物理块数量 < 逻辑块数量（说明此时需要分配新的物理块了）
        # 注：上1个推理阶段完毕后，seq的逻辑块更新了（把最新生成的这个token装进去了） 但物理块还没更新
        if len(block_table) < len(logical_blocks):
            # 物理块只允许比逻辑块少1块
            assert len(block_table) == len(logical_blocks) - 1

            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # reuse a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                # 分配一个新物理块
                new_block = self._allocate_last_physical_block(seq)
                block_table.append(new_block)
                return {}

        # 如果物理块数量 == 逻辑块数量.
        last_block = block_table[-1] # 取出最后一个物理块
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # 如果启动了prefix caching
            if self.enable_caching:
                # 检查下当前物理块slots是否被填满：
                # - 没有填满，不需要做任何操作
                # - 填满时，我们对它计算new_hash值，并根据new_hash寻找是否有可复用的物理块，
                #   若有，则释放当前物理块，取而代之以旧物理块
                maybe_new_block = self._maybe_promote_last_block(
                    seq, last_block)
                block_table[-1] = maybe_new_block
            return {}
        else:
            # 如果最后一个物理块的引用数量为 > 1 (有别的逻辑块在引用它)
            # 触发copy-on-write机制，分配一个新的物理块
            new_block = self._allocate_last_physical_block(seq)

            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return {last_block.block_number: [new_block.block_number]}

    def fork(self, parent_seq: Request, child_seq: Request) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        # When using a sliding window, blocks will be eventually reused.
        # In this case the block tables will contain repeated blocks.
        # When forking, we must make sure that each block's `ref_count`
        # is only incremented by one, so we deduplicate them by wrapping
        # them in a set.
        for block in set(src_block_table):
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: Task) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self,
                    seq_group: Task,
                    num_lookahead_slots: int = 0) -> AllocStatus:
        assert (num_lookahead_slots == 0
                ), "BlockSpaceManagerV1 does not support lookahead allocation"
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=RequestStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        if self.gpu_allocator.get_num_total_blocks() < num_required_blocks:
            return AllocStatus.NEVER
        elif num_free_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def swap_in(self,
                seq_group: Task,
                num_lookahead_slots: int = 0) -> Dict[int, int]:
        assert (num_lookahead_slots == 0
                ), "BlockSpaceManagerV1 does not support lookahead allocation"

        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=RequestStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate(
                        cpu_block.block_hash, cpu_block.num_hashed_tokens)
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: Task) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: Task) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=RequestStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate(
                        gpu_block.block_hash, gpu_block.num_hashed_tokens)
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        # when using a sliding window, each seq will only use up
        # to `self.block_sliding_window` blocks. When freeing
        # the block table, we must make sure to not free blocks more
        # than once. If no sliding window is used, there is no block
        # reuse in the block table, so we must free all blocks.
        blocks_to_free = (block_table[-self.block_sliding_window:]
                          if self.block_sliding_window is not None else
                          block_table)
        for block in set(blocks_to_free):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Request) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Request) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

    def access_all_blocks_in_seq(
        self,
        seq: Request,
        access_time: float,
    ) -> None:
        if self.enable_caching:
            # Update the last accessed time of all the blocks accessed
            # in this step.
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                block.last_accessed = access_time

    def compute_full_blocks_in_seq(self, seq: Request):
        if seq.seq_id not in self.block_tables:
            return
        max_full_block = seq.get_len() // self.block_size - 1
        block_table = self.block_tables[seq.seq_id]
        if max_full_block == -1:
            return
        for i in reversed(range(max_full_block)):
            if block_table[i].computed:
                break
            block_table[i].computed = True

    def get_all_computed_blocks(self, seq: Request) -> List[int]:
        if seq.seq_id not in self.block_tables:
            return []
        block_table = self.block_tables[seq.seq_id]
        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.
        return [
            b.block_number
            for b in takewhile(lambda b: b.computed, block_table[:-1])
        ]

    def get_common_computed_block_ids(
            self, seqs: List[Request]) -> GenericSequence[int]:
        """Return the block ids that are common for a given sequence group.

        Used in prefill (can skip prefill of some blocks).
        """
        # Can return non-empty result only with prefix caching enabled.
        if not self.enable_caching:
            return []

        ids_list = [self.get_all_computed_blocks(seq) for seq in seqs]
        return commonprefix([ids for ids in ids_list if ids != []])

    def mark_blocks_as_computed(self, seq_group: Task):
        if self.enable_caching:
            for seq in seq_group.seqs_dict.values():
                self.compute_full_blocks_in_seq(seq)
