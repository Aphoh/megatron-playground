# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
import megatron.core.activations as mact
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)


@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class GateLossScaler(torch.autograd.Function):
    loss_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, loss: torch.Tensor):
        ctx.save_for_backward(loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (loss,) = ctx.saved_tensors
        scaled_loss_grad = torch.ones_like(loss) * GateLossScaler.loss_scale
        return grad_output, scaled_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        GateLossScaler.loss_scale = scale


@torch.compile
def compute_eff_loss(logits: torch.Tensor, config: TransformerConfig):
    return logits.sigmoid().square().mean()


@torch.compile
def group_entropy_loss(logits: torch.Tensor, config: TransformerConfig):
    s, b, _ = logits.shape
    logits = logits.view(s, b, -1, config.dsparse_block_width)
    entropy_loss = torch.log_softmax(logits, dim=-1).mul(torch.softmax(logits, dim=-1)).sum()
    return entropy_loss


GATE_LOSSES = {"eff": compute_eff_loss, "group_entropy": group_entropy_loss}


class MLPActivation(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)

    def apply_gate_loss(self, intermediate_parallel: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=intermediate_parallel.device)
        for loss_type, coeff in zip(self.config.gate_aux_losses, self.config.gate_aux_loss_coeffs):
            loss += coeff * GATE_LOSSES[loss_type](intermediate_parallel, self.config)
        return GateLossScaler.apply(intermediate_parallel, loss)

    def forward(
        self, intermediate_parallel: torch.Tensor, bias_parallel: torch.Tensor
    ) -> torch.Tensor:
        should_apply_eff_loss = self.training and len(self.config.gate_aux_losses) > 0
        kwargs = {"a": self.config.swash_alpha} if self.config.activation_func == mact.swash else {}
        if self.config.bias_activation_fusion:
            fused_fn = mact.get_fused_bias_act(
                self.config.activation_func, self.config.gated_linear_unit
            )
            assert fused_fn is not None, "Could not find fused function for bias activation fusion"
            assert self.config.add_bias_linear, "Bias fusion requires add_bias_linear"
            assert not should_apply_eff_loss, "Eff loss not supported with bias fusion"
            assert bias_parallel is not None, "Bias fusion requires bias"
            intermediate_parallel = fused_fn(intermediate_parallel, bias_parallel, **kwargs)
        elif should_apply_eff_loss:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    x0 = self.apply_gate_loss(x[0])
                    return self.config.activation_func(x0, **kwargs) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.apply_gate_loss(intermediate_parallel)
                intermediate_parallel = self.config.activation_func(intermediate_parallel, **kwargs)
        else:
            fused_fn = mact.get_fused_act(
                self.config.activation_func, self.config.gated_linear_unit
            )
            assert fused_fn is not None, "Could not find function for activation"
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = fused_fn(intermediate_parallel, **kwargs)

        return intermediate_parallel


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: int = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size is not None else self.config.hidden_size

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
        )

        self.activation_func = MLPActivation(config)

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel, bias_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                sub_sd = self._sharded_state_dict_for_glu(
                    name, module, prefix, sharded_offsets, metadata
                )
            else:
                sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def _sharded_state_dict_for_glu(
        self,
        module_name: str,
        module: torch.nn.Module,
        prefix: str,
        sharded_offsets: Tuple[Tuple[int, int, int]],
        metadata: Optional[dict] = None,
    ):
        assert module_name == 'linear_fc1', module_name
        sharded_state_dict = module.sharded_state_dict(
            f'{prefix}{module_name}.', sharded_offsets, metadata
        )
        weight_key = f'{prefix}{module_name}.weight'
        prev_sh_ten = sharded_state_dict[weight_key]

        # We must split the tensor into 2 parts, each sharded separately.
        # This requires a ShardedTensorFactory which `chunk`s during saving
        # and `cat`s during loading
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        tp_shard_axis = 0
        prepend_axis_num = len(sharded_offsets)

        def sh_ten_build_fn(key: str, t: torch.Tensor, replica_id: ReplicaId):
            offset_w = (tp_shard_axis + prepend_axis_num, tp_rank, tp_size * 2)
            offset_v = (tp_shard_axis + prepend_axis_num, tp_size + tp_rank, tp_size * 2)
            with torch.no_grad():
                tensor_w, tensor_v = torch.chunk(t, 2, dim=tp_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_w,
                    *sharded_offsets,
                    offset_w,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_v,
                    *sharded_offsets,
                    offset_v,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
            ]

        def sh_ten_merge_fn(sub_state_dict):
            with torch.no_grad():
                return torch.cat(sub_state_dict)

        sharded_state_dict[weight_key] = ShardedTensorFactory(
            prev_sh_ten.key,
            prev_sh_ten.data,
            sh_ten_build_fn,
            sh_ten_merge_fn,
            prev_sh_ten.replica_id,
        )
        return sharded_state_dict


class RouterLossScaler(torch.autograd.Function):
    loss_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, loss: torch.Tensor):
        ctx.save_for_backward(loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (loss,) = ctx.saved_tensors
        scaled_loss_grad = torch.ones_like(loss) * RouterLossScaler.loss_scale
        return grad_output, scaled_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        RouterLossScaler.loss_scale = scale


class MLPDShard(MLP):
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules, is_expert: bool = False
    ):
        if config.gated_linear_unit:
            raise ValueError("MLPDShard does not support gated linear units")

        super().__init__(config, submodules, is_expert=is_expert)

        self.num_exps = self.config.ffn_hidden_size // self.config.dsparse_block_width
        self.block_width = self.config.dsparse_block_width
        assert self.block_width * self.num_exps == self.config.ffn_hidden_size
        self.router_weight = torch.nn.Parameter(
            torch.empty((self.num_exps, self.config.hidden_size))
        )
        with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
            self.config.dsparse_router_init_method(self.router_weight)

        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            raise ValueError("MLPDShard does not support tensor parallelism")

    def forward(self, hidden_states):
        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        gate_out = None
        if self.config.gated_linear_unit:
            gate_out = intermediate_parallel.chunk(2, dim=-1)[1]
            if self.config.add_bias_linear:
                gate_out += bias_parallel.chunk(2, dim=-1)[1]

        router_logits = torch.nn.functional.linear(hidden_states, self.router_weight).view(
            -1, self.num_exps
        )
        router_labels = intermediate_parallel.view(-1, self.block_width)
        router_labels = torch.any(router_labels > 0, dim=-1)
        router_preds = router_logits > 0

        true_pos = (router_preds & router_labels).sum()
        false_pos = (router_preds & ~router_labels).sum()
        false_neg = (~router_preds & router_labels).sum()
        self.accumulate_loggable("precision", num=true_pos, denom=true_pos + false_pos)
        self.accumulate_loggable("recall", num=true_pos, denom=true_pos + false_neg)

        router_loss = (
            self.config.dsparse_router_loss_coeff
            * torch.nn.functional.binary_cross_entropy_with_logits(router_logits, router_labels)
        )

        intermediate_parallel = RouterLossScaler.apply(intermediate_parallel, router_loss)

        intermediate_parallel = self.activation_func(intermediate_parallel, bias_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        return output, output_bias
