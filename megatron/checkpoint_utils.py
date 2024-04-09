from .utils import print_rank_0


def convert_pythia(state_dict: dict):
    res = {}
    print_rank_0("Converting Pythia checkpoint")

    max_layer_index = max(
        [int(key.split('.')[2]) for key in state_dict.keys() if key.startswith('gpt_neox.layers.')]
    )
    num_layers = max_layer_index + 1
    layer_maps = [
        (
            'gpt_neox.layers.%d.input_layernorm.weight',
            'decoder.layers.%d.self_attention.linear_qkv.layer_norm_weight',
        ),
        (
            'gpt_neox.layers.%d.input_layernorm.bias',
            'decoder.layers.%d.self_attention.linear_qkv.layer_norm_bias',
        ),
        (
            'gpt_neox.layers.%d.post_attention_layernorm.weight',
            'decoder.layers.%d.mlp.linear_fc1.layer_norm_weight',
        ),
        (
            'gpt_neox.layers.%d.post_attention_layernorm.bias',
            'decoder.layers.%d.mlp.linear_fc1.layer_norm_bias',
        ),
        # TODO: what is this? ('gpt_neox.layers.%d.attention.bias', 'decoder.layers.%d.self_attention.linear_qkv.bias' ),
        # TODO: what is this? ('gpt_neox.layers.%d.attention.masked_bias', ''),
        # TODO: what is this? ('gpt_neox.layers.%d.attention.rotary_emb.inv_freq', ),
        (
            'gpt_neox.layers.%d.attention.query_key_value.weight',
            'decoder.layers.%d.self_attention.linear_qkv.weight',
        ),
        (
            'gpt_neox.layers.%d.attention.query_key_value.bias',
            'decoder.layers.%d.self_attention.linear_qkv.bias',
        ),
        (
            'gpt_neox.layers.%d.attention.dense.weight',
            'decoder.layers.%d.self_attention.linear_proj.weight',
        ),
        (
            'gpt_neox.layers.%d.attention.dense.bias',
            'decoder.layers.%d.self_attention.linear_proj.bias',
        ),
        ('gpt_neox.layers.%d.mlp.dense_h_to_4h.weight', 'decoder.layers.%d.mlp.linear_fc1.weight'),
        ('gpt_neox.layers.%d.mlp.dense_h_to_4h.bias', 'decoder.layers.%d.mlp.linear_fc1.bias'),
        ('gpt_neox.layers.%d.mlp.dense_4h_to_h.weight', 'decoder.layers.%d.mlp.linear_fc2.weight'),
        ('gpt_neox.layers.%d.mlp.dense_4h_to_h.bias', 'decoder.layers.%d.mlp.linear_fc2.bias'),
    ]
    maps = [
        ('gpt_neox.final_layer_norm.weight', 'decoder.final_layernorm.weight'),
        ('gpt_neox.final_layer_norm.bias', 'decoder.final_layernorm.bias'),
    ]
    vocabs = [
        ('gpt_neox.embed_in.weight', 'embedding.word_embeddings.weight'),
        ('embed_out.weight', 'output_layer.weight'),
    ]
    for j in range(num_layers):
        for a, b in layer_maps:
            res[b % j] = state_dict.pop(a % j).float()
    for a, b in maps:
        res[b] = state_dict.pop(a).float()
    for a, b in vocabs:
        # TODO: the padding should change if tensor parallelism is used
        res[b] = state_dict.pop(a).float()[:50304] # vocab padding
    remaining_keys = list(state_dict)
    if remaining_keys:
        print_rank_0(f"Unconverted keys: {state_dict.keys()}")
    return res


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
