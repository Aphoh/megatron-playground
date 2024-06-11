import torch
import torch.nn.functional as F


@torch.compile
def swash(x, a: float = 1.0):
    return x * F.sigmoid(a * x)


@torch.compile
def bias_swash(x, b, a: float = 1.0):
    return x * F.sigmoid(a * x + b)


@torch.compile
def swaglu(x, a: float = 1.0):
    y_1, y_2 = torch.chunk(x, 2, -1)
    return swash(y_1, a) * y_2


@torch.compile
def bias_swaglu(x, b, a: float = 1.0):
    x = x + b
    y_1, y_2 = torch.chunk(x, 2, -1)
    return swash(y_1, a) * y_2
