from typing import Optional, Tuple, Union, Type
import torch
import sinfer
from sinfer.logger import init_logger
from sinfer.Intercept.mlu_hijack_utils import InterceptObject
from bangtransformer.torch import bt_ops
from sinfer.model_executor.layers.rotary_embedding import RotaryEmbedding, LinearScalingRotaryEmbedding

logger = init_logger(__name__)


@classmethod
def intercept_set_cu_seq_len(cls, cu_seq_len: torch.Tensor):
    """Compute the cos and sin cache."""
    cls.cu_seq_len = cu_seq_len
    return


@classmethod
def intercept_unset_cu_seq_len(cls):
    """Compute the cos and sin cache."""
    cls.cu_seq_len = None
    return


@classmethod
def intercept_set_max_seq_len(cls, max_seq_len: int):
    """Set max_seq_len."""
    cls.max_seq_len = max_seq_len
    return


@classmethod
def intercept_unset_max_seq_len(cls):
    """Unset max_seq_len."""
    cls.max_seq_len = None
    return


@classmethod
def intercept_set_is_prompt(cls, is_prompt: bool):
    """Set is_prompt."""
    cls.is_prompt = is_prompt
    return


@classmethod
def intercept_unset_is_prompt(cls):
    """Unset is_prompt."""
    cls.is_prompt = None
    return


@classmethod
def intercept_set_is_chunked(cls, is_chunked: bool):
    """Set is_chunked."""
    cls.is_chunked = is_chunked
    return


@classmethod
def intercept_unset_is_chunked(cls):
    """Unset is_chunked."""
    cls.is_chunked = None
    return


@classmethod
def intercept_unset_cos_sin(cls):
    """Compute the cos and sin cache."""
    cls.set_cos_sin = False
    cls.cos_ = None
    cls.sin_ = None
    return


def intercept_compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
    """Compute the inverse frequency."""
    # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
    # However, we use `torch.arange(..., dtype=torch.float)` instead to
    # avoid numerical issues with large base values (e.g., 10000000).
    # This may cause a slight numerical difference between the HF
    # implementation and ours.
    # NOTE(woosuk): To exactly match the HF implementation, we need to
    # use CPU to compute the cache and then move it to GPU. However, we
    # create the cache on GPU for faster initialization. This may cause
    # a slight numerical difference between the HF implementation and ours.
    '''
    @brief: modify inv_freq to fit bt_ops.apply_rotary()
    '''
    half_dim = self.rotary_dim // 2
    if self.is_neox_style:
        inv_freq = 1.0 / (base ** ((torch.arange(
            0, self.rotary_dim, 1, dtype=torch.float32, device="cuda") % half_dim) * 2 / self.rotary_dim)
        )
    else:
        inv_freq = 1.0 / (
                base
                ** ( torch.arange(0, self.rotary_dim, 1, device="cuda", dtype=torch.float32) // 2 * 2
                    / self.rotary_dim
                )
        )
    '''
    ==================
    '''
    return inv_freq


def intercept_RotaryEmbedding_compute_cos_sin_cache(self) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = self._compute_inv_freq(self.base)
    '''
    @brief: move 't' to mlu, make sure 't' and 'inv_freq' are both at mlu when do einsum
    '''
    t = torch.arange(self.max_position_embeddings, dtype=torch.float, device="cuda")
    '''
    ==================
    '''
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache

def intercept_LinearScalingRotaryEmbedding_compute_cos_sin_cache(self) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = self._compute_inv_freq(self.base)
    cache_list = []
    for scaling_factor in self.scaling_factors:
        # NOTE(woosuk): self.max_position_embeddings is the original
        # maximum length before applying the rope scaling.
        # Thus, the maximum length after applying the rope scaling is
        # self.max_position_embeddings * self.scaling_factor.
        max_len = self.max_position_embeddings * scaling_factor
        '''
        @brief: move 't' to mlu, make sure 't' and 'inv_freq' are both at mlu when do einsum
        '''
        t = torch.arange(max_len, dtype=torch.float, device="cuda")
        '''
        ==================
        '''
        t = t / scaling_factor

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        cache_list.append(cache)
    return torch.cat(cache_list, dim=0)

def intercept_get_cos_sin(
    self,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
    sin = sin.view(-1, self.rotary_dim)
    cos = cos.view(-1, self.rotary_dim)
    return cos, sin


def intercept_RotaryEmbedding_forward(
    self,
    positions: torch.Tensor,
    x: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
    # ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    '''
    @brief: use bt_ops.apply_rotary to replace ops.rotary_embedding
    '''
    if offsets is not None:
        raise ValueError(f"bt_ops.apply_rotary not support offsets yet.")
    else:
        if RotaryEmbedding.set_cos_sin == False:
            RotaryEmbedding.cos_, RotaryEmbedding.sin_ = self._get_cos_sin(positions)
            RotaryEmbedding.set_cos_sin = True
        interleaved = True
        if self.is_neox_style:
            interleaved = False
        if RotaryEmbedding.is_chunked or not RotaryEmbedding.is_prompt :
            position_ids = positions
            discrete = True
        else :
            position_ids = None
            discrete = False
        x = bt_ops.apply_rotary(x,
                                RotaryEmbedding.sin_,
                                RotaryEmbedding.cos_,
                                position_ids,
                                RotaryEmbedding.cu_seq_len,
                                interleaved,
                                discrete,
                                False,
                                RotaryEmbedding.max_seq_len)
    return x


RotaryEmbedding.cos_ = None
RotaryEmbedding.sin_ = None
RotaryEmbedding.cu_seq_len_ = None
RotaryEmbedding.set_cos_sin = False
RotaryEmbedding.max_seq_len = None


InterceptObject.add_intercept(RotaryEmbedding,
                                  "_set_cu_seq_len",
                                  intercept_set_cu_seq_len)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "unset_cu_seq_len",
                                  intercept_unset_cu_seq_len)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "_set_max_seq_len",
                                  intercept_set_max_seq_len)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "unset_max_seq_len",
                                  intercept_unset_max_seq_len)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "_set_is_prompt",
                                  intercept_set_is_prompt)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "_set_is_chunked",
                                  intercept_set_is_chunked)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "unset_is_prompt",
                                  intercept_unset_is_prompt)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "unset_is_chunked",
                                  intercept_unset_is_chunked)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "unset_cos_sin",
                                  intercept_unset_cos_sin)
InterceptObject.add_intercept(RotaryEmbedding,
                                  "_get_cos_sin",
                                  intercept_get_cos_sin)
InterceptObject.add_intercept(RotaryEmbedding,
                                  RotaryEmbedding._compute_inv_freq,
                                  intercept_compute_inv_freq)
InterceptObject.add_intercept(RotaryEmbedding,
                                  RotaryEmbedding._compute_cos_sin_cache,
                                  intercept_RotaryEmbedding_compute_cos_sin_cache)
InterceptObject.add_intercept(RotaryEmbedding,
                                  RotaryEmbedding.forward,
                                  intercept_RotaryEmbedding_forward)
InterceptObject.add_intercept(LinearScalingRotaryEmbedding,
                                  LinearScalingRotaryEmbedding._compute_cos_sin_cache,
                                  intercept_LinearScalingRotaryEmbedding_compute_cos_sin_cache)
