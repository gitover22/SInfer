import torch
import sinfer
from torch import nn
from typing import List, Optional, Tuple
from sinfer.attention import AttentionMetadata
from sinfer.Intercept.mlu_definitions import *
from sinfer.Intercept.intercept_utils import InterceptObject
from sinfer.logger import init_logger

from sinfer.model_executor.models.qwen import QWenAttention, QWenBlock, QWenModel
from sinfer.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2Model
from sinfer.model_executor.models.qwen2_moe import Qwen2MoeModel
from sinfer.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaModel
from sinfer.model_executor.models.falcon import FalconAttention, FalconDecoderLayer, FalconModel
from sinfer.model_executor.models.baichuan import BaiChuanAttention, BaiChuanDecoderLayer, BaiChuanModel
from sinfer.model_executor.models.chatglm import GLMAttention, GLMBlock, ChatGLMModel
from sinfer.model_executor.models.gpt_neox import GPTNeoXAttention, GPTNeoXLayer, GPTNeoXModel
from sinfer.model_executor.models.mixtral import MixtralAttention, MixtralModel
from sinfer.Intercept.model_executor.custom_model.custom import CustomModel
from sinfer.model_executor.models.bloom import BloomModel

logger = init_logger(__name__)


KVCache = Tuple[torch.Tensor, torch.Tensor]


QwenModel__forward_ori = QWenModel.forward
Qwen2Model__forward_ori = Qwen2Model.forward
Qwen2MoeModel__forward_ori = Qwen2MoeModel.forward
LlamaModel__forward_ori = LlamaModel.forward
FalconModel__forward_ori = FalconModel.forward
BaiChuanModel__forward_ori = BaiChuanModel.forward
ChatGLMModel__forward_ori = ChatGLMModel.forward
GPTNeoXModel__forward_ori = GPTNeoXModel.forward
MixtralModel__forward_ori = MixtralModel.forward
CustomModel__forward_ori = CustomModel.forward
BloomModel__forward_ori = BloomModel.forward


model_list = [
    QWenModel,
    Qwen2Model,
    Qwen2MoeModel,
    LlamaModel,
    FalconModel,
    BaiChuanModel,
    ChatGLMModel,
    GPTNeoXModel,
    MixtralModel,
    CustomModel,
    BloomModel,
]


def get_model_with_dict(model):
    model_dict = {
        "QWenModel" : QwenModel__forward_ori,
        "Qwen2Model" : Qwen2Model__forward_ori,
        "Qwen2MoeModel" : Qwen2MoeModel__forward_ori,
        "LlamaModel" : LlamaModel__forward_ori,
        "FalconModel" : FalconModel__forward_ori,
        "BaiChuanModel" : BaiChuanModel__forward_ori,
        "ChatGLMModel" : ChatGLMModel__forward_ori,
        "GPTNeoXModel" : GPTNeoXModel__forward_ori,
        "MixtralModel" : MixtralModel__forward_ori,
        "CustomModel" : CustomModel__forward_ori,
        "BloomModel" : BloomModel__forward_ori,
    }
    return model_dict.get(model.__name__)


# init layer params before model forward
def model_forward_preprocess(
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata):
    """
    is_chunked:       chunked mode rotary_emb use position_ids but not cu_seq_lens
                      True if enable_chunked_prefil is open and
                      current is not the first chunked of prefill.
    is_prompt:        if forward has prefill
    max_seq_len:      max_seq_len used by unpaged mode attn
    cu_seq_lens:      used in rotary_emb when un_chunked mode or decoder mode
    rope_max_seq_len: max_emb_len use in rotary_emb.
    """
    is_chunked = False
    is_prompt = False
    if attn_metadata.prefill_metadata:
        cu_seq_lens = attn_metadata.prefill_metadata.subquery_start_loc
        max_seq_len = attn_metadata.prefill_metadata.max_seq_len
        rope_max_seq_len = attn_metadata.prefill_metadata.max_query_len
        is_prompt = True
        if attn_metadata.decode_metadata or \
               torch.ne(attn_metadata.prefill_metadata.subquery_start_loc, \
                        attn_metadata.prefill_metadata.seq_start_loc).any():
            is_chunked = True
    if attn_metadata.decode_metadata:
        # input_ids is pack mode, and the decode_seq_len = 1
        cu_seq_lens = torch.arange(0, input_ids.shape[0] + 1, 1, dtype=torch.int32, device="cuda")
        if attn_metadata.prefill_metadata:
            rope_max_seq_len = max(rope_max_seq_len,
                attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens)
        else:
            rope_max_seq_len = 1
        max_seq_len = 1
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding._set_cu_seq_len(cu_seq_lens)
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding._set_max_seq_len(rope_max_seq_len)
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding._set_is_prompt(is_prompt)
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding._set_is_chunked(is_chunked)
    slot_mapping_new = attn_metadata.slot_mapping.flatten() % BlockSizeInfo.BLOCK_SIZE
    batch_ids = attn_metadata.slot_mapping.flatten()[cu_seq_lens[:-1]] // BlockSizeInfo.BLOCK_SIZE
    attn_metadata.cu_seq_lens = cu_seq_lens
    attn_metadata.max_seq_len = max_seq_len
    attn_metadata.slot_mapping_new = None if attn_metadata.prefill_metadata else slot_mapping_new
    attn_metadata.batch_ids = batch_ids
    return

# clear layer params after model forward
def model_forward_postprocess():
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.unset_cos_sin()
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.unset_cu_seq_len()
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.unset_max_seq_len()
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.unset_is_prompt()
    sinfer.model_executor.layers.rotary_embedding.RotaryEmbedding.unset_is_chunked()
    return


def intercept_model_forward(self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None):
    model = self.__class__
    # Now support pre/postprocess for this model
    # BaichuanModel Qwen2Model QWenModel MixtralModel ChatGLMModel GPTNeoXModel LlamaModel FalconModel
    forward = get_model_with_dict(model)
    model_forward_preprocess(input_ids, attn_metadata)
    if inputs_embeds is None:
        hidden_states = forward(self, input_ids, positions, kv_caches, attn_metadata)
    else:
        hidden_states = forward(self, input_ids, positions, kv_caches, attn_metadata, inputs_embeds)
    model_forward_postprocess()
    return hidden_states


for model in model_list:
    InterceptObject.add_intercept(model,
                                      model.forward,
                                      intercept_model_forward)
