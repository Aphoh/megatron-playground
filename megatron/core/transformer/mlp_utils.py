import torch
from megatron.core.parallel_state import _MLP_LOGGING_TRACKER
from megatron.core import parallel_state
from abc import ABC
from torchist import histogram
import wandb

@torch.compile
def eff_loss(logits: torch.Tensor, config):
    return logits.sigmoid().square().mean()


@torch.compile
def group_entropy_loss(logits: torch.Tensor, config):
    s, b, _ = logits.shape
    logits = logits.view(s, b, -1, config.dsparse_block_width)
    entropy_loss = torch.log_softmax(logits, dim=-1).mul(torch.softmax(logits, dim=-1)).sum()
    return entropy_loss


GATE_LOSSES = {"eff": eff_loss, "group_entropy": group_entropy_loss}


class BaseMetric(ABC):
    def __init__(self, name, num_layers):
        self.num_layers = num_layers
        self.name = name

    def update_local(self, *args):
        pass
    def reduce_and_get(self):
        pass
    def clear(self):
        pass
    def write_and_clear(self, writer, wandb_writer, iteration):
        pass

class NumDenomMetric(BaseMetric):
    def __init__(self, name, num_layers):
        super().__init__(name, num_layers)
        self.num = None
        self.denom = None

    @torch.no_grad()
    def update_local(self, num, denom, layer_number: int):
        if self.num is None:
            self.num = torch.zeros(self.num_layers, device=num.device)
        if self.denom is None:
            self.denom = torch.zeros(self.num_layers, device=denom.device)
        self.num[layer_number-1] += num
        self.denom[layer_number-1] += denom 

    @torch.no_grad()
    def reduce_and_get(self):
        torch.distributed.all_reduce(self.num, group=parallel_state.get_pipeline_model_parallel_group())
        torch.distributed.all_reduce(self.denom, group=parallel_state.get_pipeline_model_parallel_group())
        return self.num / self.denom

    def clear(self):
        self.num.zero_()
        self.denom.zero_()

    def write_and_clear(self, writer, wandb_writer, iteration):
        by_layer = self.reduce_and_get()
        mean = by_layer.mean()
        if writer:
            writer.add_scalar(f'mlp/{self.name}', mean, iteration)
            for i in range(self.num_layers):
                writer.add_scalar(f'mlp/{self.name}/layer_{i+1}', by_layer[i], iteration)

        if wandb_writer:
            wandb_writer.log({f'mlp/{self.name}': mean} + {
                f'mlp/{self.name}/layer_{i+1}': by_layer[i] for i in range(self.num_layers)
            }, iteration)
        self.clear()


class HistMetric(BaseMetric):
    def __init__(self, name, num_layers, bins):
        super().__init__(name, num_layers)
        self.bins = bins
        self.counts = None
        gt0_idx = torch.argwhere(bins == 0.0)
        assert gt0_idx.numel() == 1, f"Bins must contain one zero element, got {bins}"
        self.gt0_idx = gt0_idx.squeeze().item()
        assert bins[self.gt0_idx].item() == 0

    @torch.no_grad()
    def update_local(self, values, layer_number):
        if self.counts is None:
            self.counts = torch.zeros(self.num_layers, self.bins.numel() - 1, device=values.device)
        counts = histogram(values, edges=self.bins)
        self.counts[layer_number-1] += counts

    @torch.no_grad()
    def reduce_and_get(self):
        torch.distributed.all_reduce(self.counts, group=parallel_state.get_pipeline_model_parallel_group())
        return self.counts

    def clear(self):
        self.counts.zero_()

    def write_and_clear(self, writer, wandb_writer, iteration):
        by_layer = self.reduce_and_get()
        mean = by_layer.mean(dim=0)
        gt0 = by_layer[:, self.gt0_idx:].sum(dim=-1) / by_layer.sum(dim=-1)
        mean_gt0 = gt0.mean()
        if writer:
            writer.add_scalar(f"{self.name}/gt0", mean_gt0, iteration)
            writer.add_histogram(f'{self.name}/hist', mean, iteration)
            for i in range(self.num_layers):
                writer.add_histogram(f'{self.name}/layer_{i+1}_hist', by_layer[i], iteration)
                writer.add_scalar(f"{self.name}/layer_{i+1}_gt0", gt0[i], iteration)

        if wandb_writer:
            wandb_writer.log({
                f"{self.name}/gt0": mean_gt0,
                f"{self.name}/hist": wandb.Histogram(np_histogram=(mean.cpu().numpy(), self.bins.cpu().numpy()))
            }, iteration)
            for i in range(self.num_layers):
                wandb_writer.log({
                    f"{self.name}/layer_{i+1}_hist": wandb.Histogram(np_histogram=(by_layer[i].cpu().numpy(), self.bins.cpu().numpy())),
                    f"{self.name}/layer_{i+1}_gt0": gt0[i]
                }, iteration)
             

def save_nd_metric_to_tracker(name, num, denom, layer_idx, num_layers):
    if name not in _MLP_LOGGING_TRACKER:
        _MLP_LOGGING_TRACKER[name] = NumDenomMetric(name, num_layers)
    _MLP_LOGGING_TRACKER[name].update_local(num, denom, layer_idx)

def save_hist_metric_to_tracker(name, values, layer_idx, num_layers):
    if name not in _MLP_LOGGING_TRACKER:
        bins = torch.logspace(
            -10.0, #2^-10 min
            3.0, #2^3 max
            500,
            base=2.0,
            device=values.device,
        )
        inf = torch.tensor([float("inf")], device=bins.device)
        zero = torch.tensor([0.0], device=bins.device)
        bins = torch.cat((-inf, -bins.flip(0), zero, bins, inf))
        _MLP_LOGGING_TRACKER[name] = HistMetric(name, num_layers, bins)
    _MLP_LOGGING_TRACKER[name].update_local(values, layer_idx)

def track_mlp_metrics(iteration, writer, wandb_writer):
    for metric in _MLP_LOGGING_TRACKER.values():
        metric.write_and_clear(writer, wandb_writer, iteration)