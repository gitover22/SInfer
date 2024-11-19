import pytest
import torch
import torch.nn.functional as F
import torch_mlu
from bangtransformer.torch import bt_ops


act_dict = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
}

def ref_ffn(
    hidden_states, 
    up_fc_weight, 
    up_fc_bias, 
    down_proj_weight, 
    down_proj_bias, 
    gate_up_proj_weight, 
    gate_up_proj_bias, 
    layernorm_weight, 
    layernorm_bias, 
    act_mode):
    up_output = F.linear(hidden_states, up_fc_weight, bias=up_fc_bias)
    act_output = act_dict[act_mode](up_output)
    if not gate_up_proj_weight is None:
        gate_output = F.linear(hidden_states, gate_up_proj_weight, bias=gate_up_proj_bias)
        out = F.linear(act_output * gate_output, down_proj_weight, bias=down_proj_bias)
    else:
        out = F.linear(act_output, down_proj_weight, bias=down_proj_bias)
    return out

BATCH_SIZE = [1]
SEQ_LENS = [1, 64, 1024]
HIDDEN_SIZE = [16, 24]
INTER_SIZE = [32]
DTYPES = [torch.half, torch.float]
if "3" not in torch.mlu.get_device_name(0):
    DTYPES = [torch.half, torch.float]
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE)
@pytest.mark.parametrize("inter_size", INTER_SIZE)
@pytest.mark.parametrize("act_name", ["relu", "silu"]) # gelu
@pytest.mark.parametrize("use_gate", [True])
@pytest.mark.parametrize("use_gate_bias", [False])
@pytest.mark.parametrize("use_up_bias", [False])
@pytest.mark.parametrize("use_down_bias", [False])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [0])
def test_attention_project(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    inter_size: int,
    act_name: str,
    use_gate: bool,
    use_gate_bias: bool,
    use_up_bias: bool,
    use_down_bias: bool,
    dtype: torch.dtype,
    seed : int
) -> None:
    device_id = "mlu:0"
    torch.random.manual_seed(seed)
    torch.mlu.manual_seed(seed)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device_id)
    up_proj_weight= torch.randn(inter_size, hidden_size, dtype=dtype, device=device_id)
    if use_gate:
        gate_proj_weight = torch.randn(inter_size, hidden_size,  dtype=dtype, device=device_id)
    else:
        gate_proj_weight = None
    down_proj_weight = torch.randn(hidden_size, inter_size, dtype=dtype, device=device_id)

    out = bt_ops.ffn(hidden_states,
                    up_proj_weight,
                    None,
                    down_proj_weight,
                    None,
                    gate_proj_weight,
                    None,
                    act_name)
    
    ref_out = ref_ffn(
        hidden_states,
        up_proj_weight,
        None,
        down_proj_weight,
        None,
        gate_proj_weight,
        None,
        None,
        None,
        act_name
    )

    assert torch.allclose(out, ref_out, atol=1e-1, rtol=1e-1)

