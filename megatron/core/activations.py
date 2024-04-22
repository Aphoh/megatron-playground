from megatron.core.fusions import fused_bias_gelu, fused_bias_swiglu, fused_bias_geglu
import torch.nn.functional as F

bias_swiglu = fused_bias_swiglu.bias_swiglu_impl
swiglu = fused_bias_swiglu.SwiGLUFunction.apply
silu = F.silu

bias_gelu_exact = fused_bias_gelu.bias_gelu_exact
gelu_exact = fused_bias_gelu.gelu_exact

bias_geglu_approx = fused_bias_geglu.bias_geglu_impl
bias_gelu_approx = fused_bias_gelu.bias_gelu_impl
geglu_approx = fused_bias_geglu.GeGLUFunction.apply
gelu_approx = fused_bias_gelu.gelu_approx

bias_reglu = fused_bias_swiglu.bias_reglu
reglu = fused_bias_swiglu.reglu
relu = F.relu
