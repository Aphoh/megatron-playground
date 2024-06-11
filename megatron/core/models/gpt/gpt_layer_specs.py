# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules, MLPDShard, MLPDShardSubmodules
from megatron.core.transformer.custom_layers.nonparametric_layernorm import NonParametricLayerNorm
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    nonparametric_layernorm: bool = False,
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        layernorm_in_linear=not nonparametric_layernorm,
    )
    pre_mlp_layernorm = IdentityOp
    if num_experts:
        pre_mlp_layernorm = TENorm
    elif nonparametric_layernorm:
        pre_mlp_layernorm = NonParametricLayerNorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp if not nonparametric_layernorm else NonParametricLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear
                    if not nonparametric_layernorm
                    else TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=pre_mlp_layernorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_gpt_layer_local_spec(
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    nonparametric_layernorm: bool = False,
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        layernorm_in_linear=False,
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm
            if not nonparametric_layernorm
            else NonParametricLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                    k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=FusedLayerNorm
            if not nonparametric_layernorm
            else NonParametricLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True,
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
    layernorm_in_linear: bool = True,
    dsparse: bool = False,
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        assert not (
            layernorm_in_linear and not use_te
        ), "LayerNorm in linear layer is not supported without TE."
        fc1_cls = None
        if use_te and layernorm_in_linear:
            fc1_cls = TELayerNormColumnParallelLinear
        elif use_te and not layernorm_in_linear:
            fc1_cls = TEColumnParallelLinear
        else:
            assert not use_te
            assert not layernorm_in_linear
            fc1_cls = ColumnParallelLinear
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=fc1_cls, linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return ModuleSpec(
            module=MoELayer,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
            if not moe_grouped_gemm
            else None,
        )


def get_gpt_dsparse_layer_with_transformer_engine_spec(use_te: bool = True,) -> ModuleSpec:
    mlp = _get_dsparse_mlp_module_spec(use_te=use_te)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp if use_te else FusedLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                    core_attention=TEDotProductAttention if use_te else DotProductAttention,
                    linear_proj=TERowParallelLinear if use_te else RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if use_te else FusedLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_dsparse_mlp_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLPDShard,
        submodules=MLPDShardSubmodules(
            linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc1_shard_mask=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )
