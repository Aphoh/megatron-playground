from transformers import LlamaForCausalLM, LlamaConfig
import torch
from tqdm import tqdm
from .eval import get_data_subset, eval_on_dataset
import torch.nn.functional as F
from matplotlib import pyplot as plt
import wandb
import numpy as np

# do this in a separate method so the closure copies layer_idx
def make_act_stats_hook(layer_idx, act_stats, act_fn):
    def hook(module, input, output):
        per_channel = output.reshape(-1, output.shape[-1])
        act_stats["act_pct0"][layer_idx] += (per_channel > 0).sum(dim=0).to(torch.int32)
        act_stats["act_magn"][layer_idx] += act_fn(per_channel).abs().sum(dim=0)

    return hook


@torch.no_grad()
@torch.inference_mode()
def get_act_stats(model: LlamaForCausalLM, tokenizer):
    cfg: LlamaConfig = model.config
    n_layers, n_intermediate = cfg.num_hidden_layers, cfg.intermediate_size
    act_stats = {
        "act_pct0": torch.zeros((n_layers, n_intermediate), dtype=torch.int32, device=model.device),
        "act_magn": torch.zeros(
            (n_layers, n_intermediate), dtype=torch.float32, device=model.device
        ),
    }
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hk = make_act_stats_hook(i, act_stats, layer.mlp.act_fn)
        hooks.append(layer.mlp.gate_proj.register_forward_hook(hk))
    data, _tokens_per_byte = get_data_subset("calib", tokenizer)
    _nll, n_tokens = eval_on_dataset(model, data)
    for k, v in list(act_stats.items()):
        act_stats[k] = v.float() / n_tokens
    for hook in hooks:
        hook.remove()

    # Measure vector magnitudes and vector magnitudes under the LM head
    act_stats["vec_magn"] = torch.zeros(
        (n_layers, n_intermediate), dtype=torch.float32, device=model.device
    )
    act_stats["lmvec_magn"] = torch.zeros(
        (n_layers, n_intermediate), dtype=torch.float32, device=model.device
    )
    lm_head_weight = model.lm_head.weight
    for i, layer in tqdm(enumerate(model.model.layers), desc="Measuring vector magnitudes"):
        down_weight: torch.Tensor = layer.mlp.down_proj.weight.clone().detach()
        act_stats["vec_magn"][i] = torch.linalg.vector_norm(down_weight, ord=2, dim=0)
        down_weight_norm = model.model.norm(down_weight.T)
        head_weighed = F.linear(down_weight_norm, lm_head_weight)
        act_stats["lmvec_magn"][i] = torch.linalg.vector_norm(head_weighed, ord=2, dim=-1)

    act_stats["act_vec_magn"] = act_stats["vec_magn"] * act_stats["act_magn"]
    act_stats["act_lmvec_magn"] = act_stats["lmvec_magn"] * act_stats["act_magn"]

    for k, v in act_stats.items():
        act_stats[k] = v.cpu()

    return act_stats


def log_act_stats(stats):
    assert wandb.run is not None, "wandb must be initialized"
    for k, v in stats.items():
        vals = v.cpu().numpy()

        fvals = vals.flatten()

        # Sort the data
        sorted_data = np.sort(fvals)
        # Compute the CDF values
        cdf_values = np.arange(1, len(fvals) + 1) / len(fvals)
        y_interp = np.linspace(0, 1, 100)
        x_interp = np.interp(y_interp, cdf_values, sorted_data)
        table = wandb.Table(data=list(zip(x_interp, y_interp)), columns=["value", "CDF"])

        fig, ax = plt.subplots()

        per_layer = np.sort(vals, axis=-1)
        ax.imshow(per_layer, aspect="auto")
        fig.tight_layout()
        wandb.log(
            {
                f"act_stats/{k}_cdf": wandb.plot.line(table, "value", "CDF", title=f"{k} CDF"),
                f"act_stats/{k}": wandb.Histogram(v.cpu().numpy()),
                f"act_stats/{k}_heatmap": wandb.Plotly(fig),
            }
        )