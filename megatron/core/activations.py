from megatron.core.fusions import fused_bias_gelu, fused_bias_swiglu
import torch.nn.functional as F

bias_swiglu = fused_bias_swiglu.bias_swiglu_impl
bias_gelu_approx = fused_bias_gelu.bias_gelu_impl
bias_gelu_exact = fused_bias_gelu.bias_gelu_exact
gelu_exact = fused_bias_gelu.gelu_exact
gelu_approx = fused_bias_gelu.gelu_approx
silu = F.silu
swiglu = fused_bias_swiglu.SwiGLUFunction.apply