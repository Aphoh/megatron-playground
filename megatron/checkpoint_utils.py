from .utils import print_rank_0


def convert_linear_prenorm(model, state_dict, linear_key, prenorm_key, args):
    linear_ln_weight_key = f'{linear_key}.layer_norm_weight'
    linear_ln_bias_key = f'{linear_key}.layer_norm_bias'
    prenorm_weight_key = f'{prenorm_key}.weight'
    prenorm_bias_key = f'{prenorm_key}.bias'
    model_lin_ln = any(linear_ln_weight_key in k for k in model[0].state_dict().keys())
    ckpnt_lin_ln = any(linear_ln_weight_key in k for k in state_dict['model'].keys())
    has_bias = any(
        (prenorm_bias_key in k or linear_ln_bias_key in k) for k in state_dict['model'].keys()
    )
    msd = state_dict["model"]
    # Swap them in the correct direction if they're not the same
    if model_lin_ln != ckpnt_lin_ln:
        from_key_weight = linear_ln_weight_key if ckpnt_lin_ln else prenorm_weight_key
        from_key_bias = linear_ln_bias_key if ckpnt_lin_ln else prenorm_bias_key
        to_key_weight = prenorm_weight_key if ckpnt_lin_ln else linear_ln_weight_key
        to_key_bias = prenorm_bias_key if ckpnt_lin_ln else linear_ln_bias_key
        print_rank_0(f'Converting checkpoint {"to" if model_lin_ln else "from"} *LayerNormLinear')
        for j in range(args.num_layers):
            msd[f'decoder.layers.{j}.{to_key_weight}'] = msd.pop(
                f'decoder.layers.{j}.{from_key_weight}'
            )
            if has_bias:
                msd[f'decoder.layers.{j}.{to_key_bias}'] = msd.pop(
                    f'decoder.layers.{j}.{from_key_bias}'
                )


def convert_linear_prenorms(model, state_dict, args):
    convert_linear_prenorm(model, state_dict, "mlp.linear_fc1", "pre_mlp_layernorm", args)
    convert_linear_prenorm(model, state_dict, "self_attention.linear_qkv", "input_layernorm", args)
