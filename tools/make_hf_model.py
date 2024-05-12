import sys
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import torch
import types
import enum
import argparse

class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2
    retro_encoder = 3
    retro_decoder = 4

megatron = types.ModuleType('megatron')
megatron_core = types.ModuleType('megatron.core')
megatron_core_enums = types.ModuleType('megatron.core.enums')

# Set attributes appropriately
setattr(megatron_core_enums, 'ModelType', ModelType)
setattr(megatron_core, 'enums', megatron_core_enums)
setattr(megatron, 'core', megatron_core)

# Insert the mock modules into sys.modules
sys.modules['megatron'] = megatron
sys.modules['megatron.core'] = megatron_core
sys.modules['megatron.core.enums'] = megatron_core_enums


def rewrite_state_dict(state_dict, margs):
    for k in list(state_dict.keys()):
        if "_extra_state" in k:
            del state_dict[k]
    mapper = {
        "model.embed_tokens.weight": "embedding.word_embeddings.weight",
        "model.norm.weight": "decoder.final_layernorm.weight",
    }
    layer_mapper = {
        "self_attn.o_proj.weight": "self_attention.linear_proj.weight",
        "self_attn.o_proj.bias": "self_attention.linear_proj.bias",
        "input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
        "post_attention_layernorm.weight": "mlp.linear_fc1.layer_norm_weight",
        "mlp.down_proj.weight": 'mlp.linear_fc2.weight', 
        "mlp.down_proj.bias": 'mlp.linear_fc2.bias', 
    }
    if not margs.untie_embeddings_and_output_weights:
        state_dict["lm_head.weight"] = state_dict["embedding.word_embeddings.weight"]
    else:
        assert False, "untying not supported"

    for k, v in mapper.items():
        state_dict[k] = state_dict.pop(v)

    for layer in range(margs.num_layers):
        from_prefix: str = f"decoder.layers.{layer}."
        to_prefix: str = f"model.layers.{layer}."

        for k, v in layer_mapper.items():
            state_dict[to_prefix + k] = state_dict.pop(from_prefix + v)
        
        # QKV weights
        qkv_weight = state_dict.pop(from_prefix + "self_attention.linear_qkv.weight")
        qkv_bias = state_dict.pop(from_prefix + "self_attention.linear_qkv.bias")
        tp = margs.tensor_model_parallel_size
        nh = margs.num_attention_heads // tp
        ng = (
            margs.num_query_groups if margs.group_query_attention else margs.num_attention_heads
        ) // tp
        dim = margs.kv_channels
        assert nh % ng == 0
        sizes = [dim * nh // ng, dim, dim]
        n_real_heads = sum(sizes)
        q_proj, k_proj, v_proj = qkv_weight.reshape((ng, n_real_heads, -1)).split(sizes, dim=1)
        q_proj = q_proj.reshape((margs.hidden_size, nh*dim))
        k_proj = k_proj.reshape((margs.hidden_size, ng*dim))
        v_proj = v_proj.reshape((margs.hidden_size, ng*dim))
        state_dict[to_prefix + "self_attn.q_proj.weight"] = q_proj
        state_dict[to_prefix + "self_attn.k_proj.weight"] = k_proj
        state_dict[to_prefix + "self_attn.v_proj.weight"] = v_proj

        q_proj_bias, k_proj_bias, v_proj_bias = qkv_bias.reshape((ng, n_real_heads)).split(sizes, dim=1)
        state_dict[to_prefix + "self_attn.q_proj.bias"] = q_proj_bias.flatten()
        state_dict[to_prefix + "self_attn.k_proj.bias"] = k_proj_bias.flatten()
        state_dict[to_prefix + "self_attn.v_proj.bias"] = v_proj_bias.flatten()

        # MLP weights
        fc1_weight = state_dict.pop(from_prefix + "mlp.linear_fc1.weight")
        fc1_bias = state_dict.pop(from_prefix + "mlp.linear_fc1.bias")
        gate_weight, up_proj_weight = fc1_weight.chunk(2, dim=0)
        gate_bias, up_proj_bias = fc1_bias.chunk(2, dim=0)
        state_dict[to_prefix + "mlp.gate_proj.weight"] = gate_weight
        state_dict[to_prefix + "mlp.up_proj.weight"] = up_proj_weight
        state_dict[to_prefix + "mlp.gate_proj.bias"] = gate_bias
        state_dict[to_prefix + "mlp.up_proj.bias"] = up_proj_bias

def main():
    parser = argparse.ArgumentParser(description="Convert a model to HuggingFace format")
    parser.add_argument("--model", type=str, help="Path to the model")
    parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer")
    parser.add_argument("--output", type=str, help="Path to the model output directory")
    cmd_args = parser.parse_args()
    model_path = Path(cmd_args.model)
    assert model_path.exists(), f"Model {model_path} does not exist"
    with open(model_path / "latest_checkpointed_iteration.txt") as f:
        iteration = int(f.read().strip())
    pt_file = model_path / f"iter_{iteration:07d}/mp_rank_00/model_optim_rng.pt"
    assert pt_file.exists(), f"PyTorch file {pt_file} does not exist"
    ckpt = torch.load(pt_file, map_location="cpu")
    args = ckpt["args"]

    out_path = Path(cmd_args.output)
    if out_path.exists():
        print(f"Warning: output path {out_path} already exists, overwriting")
    
    assert args.normalization == "RMSNorm", "Only RMSNorm is supported"
    assert args.glu, "Only glu is supported"
    assert not (args.add_qkv_bias and not args.add_bias_linear)
    tokenizer = AutoTokenizer.from_pretrained(cmd_args.tokenizer)
    config = LlamaConfig(
        vocab_size=args.padded_vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_query_groups if args.group_query_attention else args.num_attention_heads,
        hidden_act=args.act_fn,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.norm_epsilon,
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        rope_theta=args.rotary_base,
        attention_bias=args.add_bias_linear,
        mlp_bias=args.add_bias_linear,
    )
    model = LlamaForCausalLM(config)
    sd = ckpt["model"]
    rewrite_state_dict(sd, args)

    model.load_state_dict(sd)

    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    inputs = tokenizer("Hello, my name is jason, I'm a pretty good", return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    print("Running model")
    res = model(**inputs)
    print("Got loss", res.loss.item())


if __name__ == "__main__":
    main()