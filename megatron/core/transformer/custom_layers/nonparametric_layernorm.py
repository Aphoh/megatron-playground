from torch import nn
from megatron.core.transformer import TransformerConfig
import torch.nn.functional as F
import torch

class NonParametricLayerNorm(nn.Module):
    def __init__(self, config: TransformerConfig, hidden_size: int, eps=1e-5):
        super(NonParametricLayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # x: [b, s, h]
        orig_dtype = x.dtype
        return F.layer_norm(x.to(torch.float32), (self.hidden_size,), None, None, eps=self.eps).to(orig_dtype)

    def extra_repr(self):
        return f'hidden_size={self.hidden_size}, eps={self.eps}'