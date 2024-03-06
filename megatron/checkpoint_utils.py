from utils import print_rank_0

def convert_pythia(state_dict, args):
    assert args.untie_embeddings_and_output_weights, "Pythia requires untied embeddings and output weights. Pass --untie-embeddings-and-output-weights"
    print_rank_0("Converting Pythia checkpoint")
    layer_maps = [
        ('gpt_neox.layers.%d.input_layernorm.weight', 'decoder.layers.%d.self_attention.linear_qkv.layer_norm_weight'),
        ('gpt_neox.layers.%d.input_layernorm.bias', 'decoder.layers.%d.self_attention.linear_qkv.layer_norm_bias'),
        ('gpt_neox.layers.%d.post_attention_layernorm.weight', 'decoder.layers.%d.mlp.linear_fc1.layer_norm_weight'),
        ('gpt_neox.layers.%d.post_attention_layernorm.bias', 'decoder.layers.%d.mlp.linear_fc1.layer_norm_bias'),

        # TODO: what is this? ('gpt_neox.layers.%d.attention.bias', 'decoder.layers.%d.self_attention.linear_qkv.bias' ),
        # TODO: what is this? ('gpt_neox.layers.%d.attention.masked_bias', ''),
        # TODO: what is this? ('gpt_neox.layers.%d.attention.rotary_emb.inv_freq', ),
        ('gpt_neox.layers.%d.attention.query_key_value.weight', 'decoder.layers.%d.self_attention.linear_qkv.weight'),
        ('gpt_neox.layers.%d.attention.query_key_value.bias', 'decoder.layers.%d.self_attention.linear_qkv.bias'),
        ('gpt_neox.layers.%d.attention.dense.weight', 'decoder.layers.%d.self_attention.linear_proj.weight'),
        ('gpt_neox.layers.%d.attention.dense.bias', 'decoder.layers.%d.self_attention.linear_proj.bias'),

        ('gpt_neox.layers.%d.mlp.dense_h_to_4h.weight', 'decoder.layers.%d.mlp.linear_fc1.weight'),
        ('gpt_neox.layers.%d.mlp.dense_h_to_4h.bias', 'decoder.layers.%d.mlp.linear_fc1.bias'),
        ('gpt_neox.layers.%d.mlp.dense_4h_to_h.weight', 'decoder.layers.%d.mlp.linear_fc2.weight'),
        ('gpt_neox.layers.%d.mlp.dense_4h_to_h.bias', 'decoder.layers.%d.mlp.linear_fc2.bias'),
    ]
    maps = [
        ('gpt_neox.embed_in_weight', 'embedding.word_embeddings.weight'),
        ('gpt_neox.final_layer_norm.weight', 'decoder.final_layernorm.weight'),
        ('gpt_neox.final_layer_norm.bias', 'decoder.final_layernorm.bias'),
        ('gpt_neox.embed_out_weight', 'output_layer.weight'),
    ]
    for j in range(args.num_layers):
        for a, b in layer_maps:
            state_dict['model'][a.format(j)] = state_dict['model'].pop(b.format(j))
    for a, b in maps:
        state_dict['model'][a] = state_dict['model'].pop(b)


def check_mlp_linear_prenorm(model, state_dict, args):
    model_lin_ln = any('mlp.linear_fc1.layer_norm_weight' in k for k in model.state_dict().keys())
    ckpnt_lin_ln = any('mlp.linear_fc1.layer_norm_weight' in k for k in state_dict['model'].keys())
    has_bias = any('mlp.linear_fc1.layer_norm_bias' in k or 'pre_mlp_layernorm.bias' for k in state_dict['model'].keys())
    # Swap them in the correct direction if they're not the same
    if model_lin_ln != ckpnt_lin_ln:
        print_rank_0(f'Converting checkpoint {"to" if model_lin_ln else "from"} pre-layer norm')
        assert len(model) == 1
        for j in range(args.num_layers):
            for k in (['weight', 'bias'] if has_bias else ['weight']):
                lin_ln_key = 'decoder.layers.%d.mlp.linear_fc1.layer_norm_%s' % (j, k)
                pre_mlp_key = 'decoder.layers.%d.pre_mlp_layernorm.%s' % (j, k)
                state_dict["model"][lin_ln_key if model_lin_ln else pre_mlp_key] = state_dict["model"].pop(pre_mlp_key if model_lin_ln else lin_ln_key)