from megatron.core.fusions import fused_bias_gelu, fused_bias_swiglu, fused_bias_geglu
import torch.nn.functional as F
import torch

# Supported Activation Functions
silu = F.silu
relu = F.relu
gelu_exact = fused_bias_gelu.gelu_exact
gelu_approx = fused_bias_gelu.gelu_approx
@torch.compile
def relu_squared(x):
    return torch.pow(F.relu(x), 2)

# bias GLUs
bias_swiglu = fused_bias_swiglu.bias_swiglu_impl
bias_geglu_approx = fused_bias_geglu.bias_geglu_impl
bias_reglu = fused_bias_swiglu.bias_reglu

# bias acts
bias_gelu_approx = fused_bias_gelu.bias_gelu_impl
bias_gelu_exact = fused_bias_gelu.bias_gelu_exact

# glus
swiglu = fused_bias_swiglu.SwiGLUFunction.apply
geglu_approx = fused_bias_geglu.GeGLUFunction.apply
reglu = fused_bias_swiglu.reglu


ACTIVATIONS = {
    'relu': relu,
    'relu_squared': relu_squared,
    'silu': silu,
    'gelu_exact': gelu_exact,
    'gelu': gelu_approx,
}

def get_fused_bias_act(act, glu):
    res = {
        (relu, True): bias_reglu,
        (silu, True): bias_swiglu,
        (gelu_exact, False): bias_gelu_exact,
        (gelu_approx, False): bias_gelu_approx,
        (gelu_approx, True): bias_geglu_approx,
    }
    return res.get((act, glu), None)

def get_fused_act(act, glu):
    res = {
        (relu, False): relu,
        (relu_squared, False): relu_squared,
        (silu, False): silu,
        (gelu_exact, False): gelu_exact,
        (gelu_approx, False): gelu_approx,
        # glu fusions
        (gelu_approx, True): geglu_approx,
        (relu, True): reglu,
        (silu, True): swiglu,
    }
    return res.get((act, glu), None)