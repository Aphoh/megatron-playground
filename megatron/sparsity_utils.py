from megatron.core.transformer import TransformerConfig
import torch
from torchist import histogram

def activation_logging_hook(
    layer_idx: int, config: TransformerConfig, output_dict: dict, n_bins=500, lim=5.0
):
    bins = torch.logspace(
        -10,
        torch.log2(torch.tensor(lim)),
        n_bins // 2,
        base=2.0,
        device=torch.cuda.current_device(),
    )
    inf = torch.tensor([float("inf")], device=bins.device)
    zero = torch.tensor([0.0], device=bins.device)
    bins = torch.cat((-inf, -bins.flip(0), zero, bins, inf))
    if "bins" not in output_dict:
        output_dict["bins"] = bins
    key = f"layer_{layer_idx}_fc1_hist"
    if key not in output_dict:  # haven't logged at all, init to zeros
        output_dict[key] = torch.zeros(bins.shape[0] - 1, device=bins.device)

    @torch.inference_mode()
    def hook(_module, input, *args):
        input, bias = input
        if bias is not None:
            input = input + bias
        if config.gated_linear_unit:
            input = torch.chunk(input, 2, dim=-1)[0]
        act_hist = histogram(input, edges=bins)
        output_dict[key] += act_hist

    return hook
