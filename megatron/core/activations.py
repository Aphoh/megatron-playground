from megatron.core.fusions import fused_bias_gelu, fused_bias_swiglu, fused_bias_geglu, fused_bias_swash
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
swash = fused_bias_swash.swash


# bias GLUs
bias_swiglu = fused_bias_swiglu.bias_swiglu_impl
bias_geglu_approx = fused_bias_geglu.bias_geglu_impl
bias_reglu = fused_bias_swiglu.bias_reglu
bias_swaglu = fused_bias_swash.bias_swaglu

# bias acts
bias_gelu_approx = fused_bias_gelu.bias_gelu_impl
bias_gelu_exact = fused_bias_gelu.bias_gelu_exact
bias_swash = fused_bias_swash.bias_swash

# glus
swiglu = fused_bias_swiglu.SwiGLUFunction.apply
geglu_approx = fused_bias_geglu.GeGLUFunction.apply
reglu = fused_bias_swiglu.reglu
swaglu = fused_bias_swash.swaglu


ACTIVATIONS = {
    'relu': relu,
    'relu_squared': relu_squared,
    'silu': silu,
    'gelu_exact': gelu_exact,
    'gelu': gelu_approx,
    'swash': swash,
}

def get_fused_bias_act(act, glu):
    res = {
        (relu, True): bias_reglu,
        (silu, True): bias_swiglu,
        (gelu_exact, False): bias_gelu_exact,
        (gelu_approx, False): bias_gelu_approx,
        (gelu_approx, True): bias_geglu_approx,
        (swash, False): bias_swash,
        (swash, True): bias_swaglu,
    }
    return res.get((act, glu), None)

def get_fused_act(act, glu):
    res = {
        (relu, False): relu,
        (relu_squared, False): relu_squared,
        (silu, False): silu,
        (gelu_exact, False): gelu_exact,
        (gelu_approx, False): gelu_approx,
        (swash, False): swash,
        # glu fusions
        (gelu_approx, True): geglu_approx,
        (relu, True): reglu,
        (silu, True): swiglu,
        (swash, True): swaglu,
    }
    return res.get((act, glu), None)