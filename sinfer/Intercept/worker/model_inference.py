import os
import time
import torch
import sinfer
import numpy as np
from typing import List, Optional, Set
from sinfer.Intercept.intercept_utils import InterceptObject, make_tensor_with_pad, make_tensor_without_pad
from sinfer.Intercept.mlu_definitions import USE_PAGED, BlockSizeInfo, check_latency_debug_with_device, CUDA_GRAPH_DEBUG
from sinfer.logger import init_logger
from sinfer.distributed.device_communicators import pynccl_utils, custom_all_reduce
from sinfer.task import TaskMetadata
from sinfer.worker.model_inference import (ModelInference, CUDAGraphRunner,
                                      PreparePromptMetadata, PrepareDecodeMetadata,
                                      _get_graph_batch_size, _BATCH_SIZES_TO_CAPTURE,
                                      _BATCH_SIZE_ALIGNMENT, _PAD_SLOT_ID)
from sinfer.attention.backends.abstract import AttentionMetadata
from sinfer.attention.backends.flashinfer import FlashInferBackend
from sinfer.lora.layers import LoRAMapping
from sinfer.lora.request import LoRARequest
from sinfer.request import SamplerOutput
from sinfer.utils import get_kv_cache_torch_dtype

import sinfer.envs as envs

logger = init_logger(__name__)


def intercept_ModelInference_set_block_size(self, block_size: int) -> None:
    # support unpaged mode
    BlockSizeInfo.set_block_size(block_size)
    self.block_size = BlockSizeInfo.BLOCK_SIZE
    
    self.graph_block_tables = np.zeros(
        (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
        dtype=np.int32)


def intercept_ModelInference_get_max_block_per_batch(self) -> int:
    block_size = self.block_size
    # support unpaged mode
    if USE_PAGED:
        return (self.max_seq_len_to_capture + block_size - 1) // block_size
    return 1
    


def intercept_ModelInference_prepare_prompt(
    self,
    seq_group_metadata_list: List[TaskMetadata],
) -> PreparePromptMetadata:
    input_tokens: List[int] = []
    input_positions: List[int] = []
    slot_mapping: List[int] = []
    lora_index_mapping: List[int] = []
    lora_prompt_mapping: List[int] = []
    lora_requests: Set[LoRARequest] = set()

    seq_lens: List[int] = []
    context_lens: List[int] = []
    query_lens: List[int] = []
    prefix_block_tables: List[List[int]] = []
    multi_modal_input_list: List[torch.Tensor] = []

    if len(seq_group_metadata_list) == 0:
        return PreparePromptMetadata.empty()

    for seq_group_metadata in seq_group_metadata_list:
        assert seq_group_metadata.is_prompt
        seq_ids = list(seq_group_metadata.req_data.keys())
        assert len(seq_ids) == 1
        seq_id = seq_ids[0]

        computed_block_nums = seq_group_metadata.computed_block_nums
        if (self.scheduler_config is not None
                and self.scheduler_config.chunked_prefill_enabled
                and not (computed_block_nums is None
                            or computed_block_nums == [])):
            raise RuntimeError(
                "chunked prefill cannot be used with prefix caching "
                "now.")

        token_chunk_size = seq_group_metadata.token_chunk_size
        seq_data = seq_group_metadata.req_data[seq_id]
        context_len = seq_data.get_num_computed_tokens()
        # We should use get_len here because in case of preemption
        # it contains output tokens.
        seq_len = min(seq_data.get_len(), context_len + token_chunk_size)
        prompt_tokens = seq_data.get_token_ids()[context_len:seq_len]
        seq_lens.append(seq_len)

        # NOTE: This only works for oooooooxxx style attention.
        if computed_block_nums is not None and len(
                computed_block_nums) > 0 and self.sliding_window is None:
            # Prefix is not supported with sliding_window
            context_len = len(computed_block_nums) * self.block_size
            prompt_tokens = prompt_tokens[context_len:]
            prefix_block_tables.append(computed_block_nums)
        elif self.scheduler_config.chunked_prefill_enabled:
            if seq_group_metadata.block_tables is not None:
                # Prefill has chunked before.
                block_table = seq_group_metadata.block_tables[seq_id]
                prefix_block_tables.append(block_table)
            else:
                # The first prefill.
                prefix_block_tables.append([])
        else:
            prefix_block_tables.append([])
            # Right now, prefill start is always 0. However, this
            # assumption can be changed once chunked prefill is introduced.
            assert context_len == 0

        # actual prompt lens
        context_lens.append(context_len)
        query_lens.append(seq_len - context_len)

        input_tokens.extend(prompt_tokens)
        # NOTE(woosuk): Here we assume that the first token in the prompt
        # is always the first token in the sequence.
        input_positions.extend(list(range(context_len, seq_len)))
        lora_id = seq_group_metadata.lora_int_id

        if lora_id > 0:
            lora_requests.add(seq_group_metadata.lora_request)

        lora_index_mapping += [lora_id] * (seq_len - context_len)
        lora_prompt_mapping.extend(
            [lora_id] *
            (seq_len - context_len
                if seq_group_metadata.sampling_params.prompt_logprobs else 1))

        if seq_group_metadata.multi_modal_data:
            multi_modal_input_list.append(
                seq_group_metadata.multi_modal_data.data)

        if seq_group_metadata.block_tables is None:
            # During memory profiling, the block tables are not initialized
            # yet. In this case, we just use a dummy slot mapping.
            slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
            continue

        # Compute the slot mapping.
        block_table = seq_group_metadata.block_tables[seq_id]

        # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
        # where start_idx is max(0, seq_len - sliding_window).
        # For example, if the prompt len is 10, sliding window is 8, and
        # block size is 4, the first two tokens are masked and the slot
        # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
        start_idx = 0
        if self.sliding_window is not None:
            assert context_len == 0, (
                "Prefix caching is currently not supported with "
                "sliding window attention")
            start_idx = max(0, seq_len - self.sliding_window)

        for i in range(context_len, seq_len):
            if i < start_idx:
                slot_mapping.append(_PAD_SLOT_ID)
                continue

            block_number = block_table[i // self.block_size]
            block_offset = i % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)

    max_query_len = max(query_lens)
    max_seq_len = max(seq_lens)
    assert max_query_len > 0

    context_lens_tensor = torch.tensor(context_lens,
                                        dtype=torch.int,
                                        device=self.device)

    if multi_modal_input_list:
        assert self.vision_language_config, (
            "Multi-modal inputs are only supported by "
            "vision language models.")
        multi_modal_input = torch.cat(multi_modal_input_list,
                                        dim=0).to(self.device)
    else:
        multi_modal_input = None

    # Prepare prefix block tables
    max_prompt_block_table_len = max(len(t) for t in prefix_block_tables)
    '''
    @brief: use optimize make_tensor_with_pad function
    ''' 
    block_tables = make_tensor_with_pad(
        prefix_block_tables,
        max_len=max_prompt_block_table_len,
        pad=0,
        dtype=torch.int,
        device=self.device,
    )

    # Query length can be shorter than key (i.e., prompt) when prefill
    # is chunked or prefix cached.
    query_lens_tensor = torch.tensor(query_lens,
                                        dtype=torch.int32,
                                        device=self.device)
    subquery_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                        dtype=torch.int32,
                                        device=self.device)

    seq_lens_tensor = torch.tensor(seq_lens,
                                    dtype=torch.int,
                                    device=self.device)
    seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device=self.device)

    torch.cumsum(query_lens_tensor,
                    dim=0,
                    dtype=subquery_start_loc.dtype,
                    out=subquery_start_loc[1:])

    torch.cumsum(seq_lens_tensor,
                    dim=0,
                    dtype=seq_start_loc.dtype,
                    out=seq_start_loc[1:])

    if self.attn_backend is FlashInferBackend:
        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            use_cuda_graph=False,
            seq_start_loc=seq_start_loc,
            max_seq_len=max_seq_len,
            block_tables=block_tables)
    else:
        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            subquery_start_loc=subquery_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
        )

    return PreparePromptMetadata(
        input_tokens=input_tokens,
        input_positions=input_positions,
        attn_metadata=attn_metadata,
        seq_lens=seq_lens,
        query_lens=query_lens,
        lora_index_mapping=lora_index_mapping,
        lora_prompt_mapping=lora_prompt_mapping,
        lora_requests=lora_requests,
        multi_modal_input=multi_modal_input,
        slot_mapping=slot_mapping,
    )


def intercept_ModelInference_prepare_decode(
    self,
    seq_group_metadata_list: List[TaskMetadata],
) -> PrepareDecodeMetadata:
    input_tokens: List[int] = []
    input_positions: List[int] = []
    slot_mapping: List[int] = []
    seq_lens: List[int] = []
    block_tables: List[List[int]] = []
    lora_index_mapping: List[int] = []
    lora_prompt_mapping: List[int] = []
    lora_requests: Set[LoRARequest] = set()

    # The following fields are only for flashinfer
    # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
    # for the precise definition of the following fields.
    # An example:
    # request 1, page indices [0, 5, 8]
    # request 2, page indices [1, 6, 7]
    # request 3, page indices [3, 4]
    # paged_kv_indices is a concatenation of page indices of all requests:
    # [0, 5, 8, 1, 6, 7, 3, 4]
    # paged_kv_indptr is used to index into paged_kv_indices:
    # [0, 3, 6, 8]
    paged_kv_indices: List[int] = []
    # 0 at the beginning of paged_kv_indptr indicates the start of the
    # first request’s page indices in the paged_kv_indices list.
    paged_kv_indptr: List[int] = [0]
    # paged_kv_last_page_len is the length of the last page of each request
    paged_kv_last_page_len: List[int] = []

    if len(seq_group_metadata_list) == 0:
        return PrepareDecodeMetadata.empty()

    for seq_group_metadata in seq_group_metadata_list:
        assert not seq_group_metadata.is_prompt
        assert seq_group_metadata.token_chunk_size == 1

        seq_ids = list(seq_group_metadata.req_data.keys())
        lora_id = seq_group_metadata.lora_int_id

        if lora_id > 0:
            lora_requests.add(seq_group_metadata.lora_request)

        for seq_id in seq_ids:
            seq_data = seq_group_metadata.req_data[seq_id]
            generation_token = seq_data.get_last_token_id()
            input_tokens.append(generation_token)

            seq_len = seq_data.get_len()
            position = seq_len - 1
            input_positions.append(position)

            seq_len = seq_len if self.sliding_window is None else min(
                seq_len, self.sliding_window)
            seq_lens.append(seq_len)

            block_table = seq_group_metadata.block_tables[seq_id]
            block_number = block_table[position // self.block_size]
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)
            lora_index_mapping.append(lora_id)
            lora_prompt_mapping.append(lora_id)

            if self.sliding_window is not None:
                sliding_window_blocks = (self.sliding_window //
                                            self.block_size)
                block_table = block_table[-sliding_window_blocks:]
            block_tables.append(block_table)

            paged_kv_indices.extend(block_table)
            paged_kv_indptr.append(paged_kv_indptr[-1] + len(block_table))
            last_page_len = seq_data.get_len() % self.block_size
            if last_page_len == 0:
                last_page_len = self.block_size
            paged_kv_last_page_len.append(last_page_len)

    # SInfer uses cuda graph only for decoding requests.
    # See `capture_model` API for more details.
    # For decoding requests, batch_size == input_tokens.
    batch_size = len(input_tokens)
    max_seq_len = max(seq_lens)
    use_captured_graph = (not self.model_config.enforce_eager
                            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
                            and max_seq_len <= self.max_seq_len_to_capture)

    if CUDA_GRAPH_DEBUG and not self.model_config.enforce_eager and not use_captured_graph:
        logger.warning(f"Because one of the following conditions is not met, CUDA Graphs will not be enabled.\n" +
                       f"1. batch_size({batch_size}) <= max_batch_size_to_capture({_BATCH_SIZES_TO_CAPTURE[-1]})\n" +
                       f"2. max_seq_len({max_seq_len}) <= max_seq_len_to_capture({self.max_seq_len_to_capture})")

    if use_captured_graph:
        '''
        @brief: Use the refact function to get right batch size to inference
        ''' 
        graph_batch_size = \
            intercept_ModelInference_get_graph_batch_size(batch_size)
        
        assert graph_batch_size >= batch_size
        for _ in range(graph_batch_size - batch_size):
            input_tokens.append(0)
            input_positions.append(0)
            slot_mapping.append(_PAD_SLOT_ID)
            seq_lens.append(1)
            block_tables.append([])
            lora_index_mapping.append(0)
        batch_size = graph_batch_size

    seq_lens_tensor = torch.tensor(seq_lens,
                                    dtype=torch.int,
                                    device=self.device)

    if use_captured_graph:
        # When using cuda-graph all these tensors should be
        # padded.
        assert seq_lens_tensor.shape[0] == len(input_tokens)
        assert seq_lens_tensor.shape[0] == len(input_positions)
        assert seq_lens_tensor.shape[0] == len(slot_mapping)

        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        input_block_tables = self.graph_block_tables[:batch_size]
        for i, block_table in enumerate(block_tables):
            if block_table:
                input_block_tables[i, :len(block_table)] = block_table
        block_tables = torch.tensor(input_block_tables, device=self.device)
    else:
        max_block_table_len = max(
            len(block_table) for block_table in block_tables)
        # support unpaged mode, use optimize make_tensor_with_pad function
        if USE_PAGED:
            block_tables = make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )
        else:
            block_tables = make_tensor_without_pad(
                block_tables,
                dtype=torch.int,
                device=self.device,
            )

    if self.attn_backend is FlashInferBackend:
        if not hasattr(self, "flashinfer_workspace_buffer"):
            # Allocate 16MB workspace buffer
            # Follow the example of flashinfer: https://docs.flashinfer.ai/api/python/decode.html
            self.flashinfer_workspace_buffer = torch.empty(
                16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        paged_kv_indptr = torch.tensor(paged_kv_indptr,
                                        dtype=torch.int,
                                        device=self.device)
        paged_kv_indices = torch.tensor(paged_kv_indices,
                                        dtype=torch.int,
                                        device=self.device)
        paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len,
                                                dtype=torch.int,
                                                device=self.device)
        kv_cache_dtype = get_kv_cache_torch_dtype(self.kv_cache_dtype,
                                                    self.model_config.dtype)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            use_cuda_graph=False,
            workspace_buffer=self.flashinfer_workspace_buffer,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.model_config.get_num_attention_heads(
                self.parallel_config),
            num_kv_heads=self.model_config.get_num_kv_heads(
                self.parallel_config),
            head_dim=self.model_config.get_head_size(),
            page_size=self.block_size,
            data_type=kv_cache_dtype)
    else:
        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=None,
            max_seq_len=max_seq_len,
            subquery_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )
    return PrepareDecodeMetadata(
        input_tokens=input_tokens,
        input_positions=input_positions,
        attn_metadata=attn_metadata,
        lora_index_mapping=lora_index_mapping,
        lora_prompt_mapping=lora_prompt_mapping,
        lora_requests=lora_requests,
        slot_mapping=slot_mapping,
    )


@torch.inference_mode()
def intercept_ModelInference_capture_model(self, kv_caches: List[torch.Tensor], num_gpu_blocks: int) -> None:
    """Cuda graph capture a model.

    Note that CUDA graph's performance gain is negligible if number
    of batched tokens are larger than 200. And since CUDA graph
    requires fixed sized tensors, supporting large/variable batch
    size requires high GPU memory overhead. Thus, SInfer only captures
    decoding requests. Mixed batch (chunked prefill + decoding) or
    prefill requests are not captured.

    Since it is used for decoding-only, it assumes there's only 1 token
    per sequence in the batch.
    """
    # This is a hack to ensure that the NCCL backend is never deleted before the CUDA graphs.
    self.pynccl_backend = pynccl_utils.get_nccl_backend()

    assert not self.model_config.enforce_eager
    logger.info("Capturing the model for CUDA graphs. This may lead to "
                "unexpected consequences if the model is not static. To "
                "run the model in eager mode, set 'enforce_eager=True' or "
                "use '--enforce-eager' in the CLI.")
    logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                "If you are running out of memory, consider decreasing "
                "`gpu_memory_utilization` or enforcing eager mode. "
                "You can also reduce the `max_num_seqs` as needed "
                "to decrease memory usage.")
    start_time = time.perf_counter()

    '''
    @brief: add num_gpu_blocks for mlu graph capture, change torch.long to torch.int32, support unpaged mode
    '''  
    # Prepare dummy inputs. These will be reused for all batch sizes.
    global _BATCH_SIZES_TO_CAPTURE
    max_batch_size = min(max(_BATCH_SIZES_TO_CAPTURE), num_gpu_blocks)
    input_tokens = torch.zeros(max_batch_size, dtype=torch.int32).cuda()
    input_positions = torch.zeros(max_batch_size, dtype=torch.int32).cuda()
    slot_mapping = torch.empty(max_batch_size, dtype=torch.int32).cuda()
    slot_mapping.fill_(_PAD_SLOT_ID if USE_PAGED else 0)
    seq_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
    block_tables = torch.from_numpy(self.graph_block_tables).cuda()

    graph_batch_size = _get_graph_batch_size(
        self.scheduler_config.max_num_reqs)
    graph_batch_size = min(graph_batch_size, max_batch_size)
    batch_size_capture_list = [
        bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
    ]

    if graph_batch_size not in batch_size_capture_list:
        batch_size_capture_list.append(graph_batch_size)

    with custom_all_reduce.capture():
        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        for batch_size in reversed(batch_size_capture_list):
            # Create dummy attn_metadata.
            decode_metadata = self.attn_backend.make_metadata(
                is_prompt=False,
                seq_lens=None,
                seq_lens_tensor=seq_lens[:batch_size],
                max_query_len=None,
                max_seq_len=self.max_seq_len_to_capture if USE_PAGED \
                    else min(self.block_size, self.max_seq_len_to_capture),
                subquery_start_loc=None,
                seq_start_loc=None,
                context_lens_tensor=None,
                block_tables=block_tables[:batch_size],
                use_cuda_graph=True,
            )
            attn_metadata = AttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=batch_size,
                slot_mapping=slot_mapping[:batch_size],
                prefill_metadata=None,
                decode_metadata=decode_metadata,
                kv_cache_dtype=self.kv_cache_dtype,
            )

            if self.lora_config:
                lora_mapping = LoRAMapping(
                    [0] * batch_size,
                    [0] * batch_size,
                )
                self.set_active_loras(set(), lora_mapping)

            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(
                input_tokens[:batch_size],
                input_positions[:batch_size],
                kv_caches,
                attn_metadata,
                memory_pool=self.graph_memory_pool,
            )
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # This usually takes < 10 seconds.
    logger.info("Graph capturing finished in %.0f secs.", elapsed_time)

    # NOTE: Added by sinfer_mlu
    # Set the actually capture batch list to _BATCH_SIZES_TO_CAPTURE
    _BATCH_SIZES_TO_CAPTURE = batch_size_capture_list

@torch.inference_mode()
def intercept_ModelInference_execute_model(
    self,
    seq_group_metadata_list: List[TaskMetadata],
    kv_caches: List[torch.Tensor],
) -> Optional[SamplerOutput]:
    (input_tokens, input_positions, attn_metadata, sampling_metadata,
        lora_requests, lora_mapping, multi_modal_input
        ) = self.prepare_input_tensors(seq_group_metadata_list) # 准备输入张量

    if self.lora_config:
        self.set_active_loras(lora_requests, lora_mapping)

    # 目前cuda graph 只支持 decode 阶段.
    prefill_meta = attn_metadata.prefill_metadata
    decode_meta = attn_metadata.decode_metadata
    if prefill_meta is None and decode_meta.use_cuda_graph: # 如果是 decode 阶段并启用了 CUDA 图，选择与当前 batch 大小匹配的 CUDA 图执行器
        graph_batch_size = input_tokens.shape[0]
        model_executable = self.graph_runners[graph_batch_size]
    else:
        model_executable = self.model # LlamaForCausalLM
    execute_model_kwargs = {
        "input_ids": input_tokens,
        "positions": input_positions,
        "kv_caches": kv_caches,
        "attn_metadata": attn_metadata,
    }
    if self.vision_language_config:
        execute_model_kwargs.update({"image_input": multi_modal_input})

    # 插入时间戳标记
    if check_latency_debug_with_device():
        use_cuda_graph = (prefill_meta is None and decode_meta.use_cuda_graph)
        # if use_cuda_graph, the start timestamp will be inserted inside CUDAGraphRunner.forward()
        if not use_cuda_graph:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
   
    hidden_states = model_executable(**execute_model_kwargs)

    # 基于隐藏状态 hidden_states 和采样元数据 sampling_metadata，计算 logits
    logits = self.model.compute_logits(hidden_states, sampling_metadata)

    # 引入一个新的实例变量time_markers并设置
    if check_latency_debug_with_device():
        end_marker = torch.cuda.Event(enable_timing=True)
        end_marker.record()
        if use_cuda_graph:
            self.time_markers = (model_executable.start, end_marker)
        else:
            self.time_markers = (start, end_marker)

    # 只有driver worker才进行采样
    if not self.is_driver_worker:
        return None

    # 对概率分布采样下一个token
    output = self.model.sample(
        logits=logits,
        sampling_metadata=sampling_metadata,
    )

    return output


def intercept_ModelInference_get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """

    '''
    @brief: refact the '_get_graph_batch_size' function to get right batch size to inference
    '''  
    # Given batch size is larger than max capture batch size, it should never
    # happen when using mlu graph
    assert batch_size <= _BATCH_SIZES_TO_CAPTURE[-1], f"Current batch size {batch_size} \
                is larger than the max capture batch size {max(_BATCH_SIZES_TO_CAPTURE[-1])}."

    # Find the first capture batch size >= padded batch size and return it
    for capture_batch_size in _BATCH_SIZES_TO_CAPTURE:
        if capture_batch_size >= batch_size:
            return capture_batch_size

def intercept_CUDAGraphRunner_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
    **kwargs,
) -> torch.Tensor:
    # KV缓存是固定的张量，所以我们不需要复制它们。
    del kv_caches

    # Copy the input tensors to the input buffers.
    self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
    self.input_buffers["positions"].copy_(positions, non_blocking=True)
    self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                             non_blocking=True)
    self.input_buffers["seq_lens_tensor"].copy_(
        attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
    self.input_buffers["block_tables"].copy_(
        attn_metadata.decode_metadata.block_tables, non_blocking=True)
    
    if check_latency_debug_with_device():
        self.start = torch.cuda.Event(enable_timing=True)
        self.start.record()
    
    self.graph.replay()

    # Return the output tensor.
    return self.output_buffers["hidden_states"]

InterceptObject.add_intercept(ModelInference,     ModelInference.set_block_size,          intercept_ModelInference_set_block_size)
InterceptObject.add_intercept(ModelInference,     ModelInference.get_max_block_per_batch, intercept_ModelInference_get_max_block_per_batch)
InterceptObject.add_intercept(ModelInference,     ModelInference._prepare_prompt,         intercept_ModelInference_prepare_prompt)
InterceptObject.add_intercept(ModelInference,     ModelInference._prepare_decode,         intercept_ModelInference_prepare_decode)
InterceptObject.add_intercept(ModelInference,     ModelInference.capture_model,           intercept_ModelInference_capture_model)
InterceptObject.add_intercept(ModelInference,     ModelInference.infer_model,           intercept_ModelInference_execute_model)
InterceptObject.add_intercept(CUDAGraphRunner, CUDAGraphRunner.forward,             intercept_CUDAGraphRunner_forward)
