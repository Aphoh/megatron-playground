import torch
from torch.nn.parameter import Parameter
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    linear_with_grad_accumulation_and_async_allreduce,
    set_tensor_model_parallel_attributes,
    _initialize_affine_weight_gpu
)
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.transformer import MegatronModule
from megatron.core.transformer import TransformerConfig
from typing import Type


class LoraRowParallelLinear(RowParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "config" in kwargs
        self.config: TransformerConfig = kwargs["config"]
        lora_rank = self.config.lora_rank

        kwargs = dict(dtype=self.config.params_dtype, device=torch.cuda.current_device())

        if not kwargs.get("skip_weight_param_allocation", False):
            assert not self.config.use_cpu_initialization, "LoRA doesn't support CPU initialization yet"
            self.lora_a = Parameter(torch.empty(self.input_size_per_partition, lora_rank, **kwargs))
            self.lora_b = Parameter(torch.empty(lora_rank, self.output_size, **kwargs))

            _initialize_affine_weight_gpu(weight=self.lora_a, init_method=self.config.init_method, partition_dim=1)
            torch.nn.init.zeros_(self.lora_b)
        else:
            self.lora_a = None
            self.lora_b = None

        self.explicit_expert_comm = True  # TODO: rename this
        assert not self.sequence_parallel, "LoRA doesn't support sequence parallel yet"
        assert self.skip_bias_add == True or self.bias is None, "LoRA requires skip_bias_add=True, or no bias"

    def forward(self, x: torch.Tensor, *args, **kwargs):
        intermediate_parallel, bias_parallel = super().forward(x, *args, **kwargs)
        lora_parallel = linear_with_grad_accumulation_and_async_allreduce(
            x,
            self.lora_a,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
        )
        lora_parallel = torch.matmul(lora_parallel, self.lora_b.t())
        intermediate_parallel = intermediate_parallel + lora_parallel
        output_ = reduce_from_tensor_model_parallel_region(intermediate_parallel)
        return output_, bias_parallel
    

class LoraColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "config" in kwargs
        self.config: TransformerConfig = kwargs["config"]
        lora_rank = self.config.lora_rank

        kwargs = dict(dtype=self.config.params_dtype, device=torch.cuda.current_device())

        if not kwargs.get("skip_weight_param_allocation", False):
            assert not self.config.use_cpu_initialization, "LoRA doesn't support CPU initialization yet"
            self.lora_a = Parameter(torch.empty(self.input_size, lora_rank, **kwargs))
            self.lora_b = Parameter(torch.empty(lora_rank, self.output_size_per_partition, **kwargs))

            with get_cuda_rng_tracker().fork():
                self.config.init_method(self.lora_a)
            _initialize_affine_weight_gpu(weight=self.lora_b, init_method=torch.nn.init.zeros_, partition_dim=0)
        else:
            self.lora_a = None
            self.lora_b = None

        self.explicit_expert_comm = True  # TODO: rename this
        assert not self.sequence_parallel, "LoRA doesn't support sequence parallel yet"
        assert self.skip_bias_add == True or self.bias is None, "LoRA requires skip_bias_add=True, or no bias"

    def forward(self, x: torch.Tensor, *args, **kwargs):
        intermediate_parallel, bias_parallel = super().forward(x, *args, **kwargs)
        lora_parallel = torch.matmul(intermediate_parallel, self.lora_a.t())
        lora_parallel = linear_with_grad_accumulation_and_async_allreduce(
            lora_parallel,
            self.lora_b,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
        )
        output_ = intermediate_parallel + lora_parallel
        return output_, bias_parallel