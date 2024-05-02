# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
from arg_utils import ModelDescriptor, ARG_SETTERS
import safetensors.torch as sttorch
from huggingface_hub import hf_hub_download 
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError


def try_get_file(
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: str,
) -> bool:
    try:
        res = hf_hub_download(repo_id, filename, revision=revision, cache_dir=cache_dir)
        return res
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return None

def load_state_dict(args):
    weight_files = []
    for path in ["model.safetensors", "pytorch_model.bin"]:
        if (file := try_get_file(args.repo, path, revision=args.revision, cache_dir=args.hf_cache_dir)) is not None:
            weight_files.append(file)
            break
        elif (file := try_get_file(
            args.repo, path + ".index.json", revision=args.revision, cache_dir=args.hf_cache_dir
        )):
            index_contents = json.load(open(file))
            paths = list(set(index_contents["weight_map"].values()))
            for path in paths:
                weight_files.append(
                    hf_hub_download(args.repo, path, revision=args.revision, cache_dir=args.hf_cache_dir)
                )
            break
    else:
        raise ValueError("No checkpoint files found!")

    state_dict = {}
    for weight_file in weight_files:
        if weight_file.endswith(".safetensors"):
            state_dict |= sttorch.load_file(weight_file, device="cpu")
        else:
            assert weight_file.endswith(".bin")
            state_dict |= torch.load(weight_file, map_location="cpu")

    return state_dict

def add_arguments(parser):
    group = parser.add_argument_group(title='Pythia HF loader')

    group.add_argument(
        "--hf-model-type",
        type=str,
        required=True,
        choices=["pythia", "llama", "olmo"],
        help="Huggingface model type.",
    )
    group.add_argument(
        "--hf-cache-dir", type=str, default=None, help="Cache directory for Huggingface models."
    )
    group.add_argument("--repo", type=str, required=True, help="Huggingface model repository.")
    group.add_argument("--revision", type=str, default="main", help="Huggingface model revision.")
    group.add_argument(
        '--megatron-path', type=str, default=None, help='Base directory of the megatron repository'
    )

def pythia_get_tensor(state_dict, key, _margs, layer=None):
    prefix = ""
    if layer is not None:
        prefix = f"gpt_neox.layers.{layer}."
    reskey = (
        prefix
        + {
            "word embeddings": "gpt_neox.embed_in.weight",
            "input norm weight": "input_layernorm.weight",
            "input norm bias": "input_layernorm.bias",
            "post norm weight": "post_attention_layernorm.weight",
            "post norm bias": "post_attention_layernorm.bias",
            "mlp l0 weight": "mlp.dense_h_to_4h.weight",
            "mlp l0 bias": "mlp.dense_h_to_4h.bias",
            "mlp l1 weight": "mlp.dense_4h_to_h.weight",
            "mlp l1 bias": "mlp.dense_4h_to_h.bias",
            "qkv weight": "attention.query_key_value.weight",
            "qkv bias": "attention.query_key_value.bias",
            "dense weight": "attention.dense.weight",
            "dense bias": "attention.dense.bias",
            "final norm weight": "gpt_neox.final_layer_norm.weight",
            "final norm bias": "gpt_neox.final_layer_norm.bias",
            "output layer weight": "embed_out.weight",
        }[key]
    )

    return state_dict.pop(reskey)


def llama_get_tensor(state_dict, key, margs, layer=None):
    prefix = ""
    if layer is not None:
        prefix = f"model.layers.{layer}."
    mapper = {
        "word embeddings": "model.embed_tokens.weight",
        "input norm weight": "input_layernorm.weight",
        "post norm weight": "post_attention_layernorm.weight",
        "mlp l1 weight": "mlp.down_proj.weight",
        "mlp l0 weight W": "mlp.gate_proj.weight",
        "mlp l0 weight V": "mlp.up_proj.weight",
        "dense weight": "self_attn.o_proj.weight",
        "final norm weight": "model.norm.weight",
        "output layer weight": "lm_head.weight",
    }
    if key in mapper:
        return state_dict.pop(prefix + mapper[key])
    elif key == "qkv weight":
        tp = margs.tensor_model_parallel_size
        nh = margs.num_attention_heads // tp
        ng = (
            margs.num_query_groups if margs.group_query_attention else margs.num_attention_heads
        ) // tp
        dim = margs.kv_channels
        assert nh % ng == 0
        q_proj = state_dict.pop(prefix + "self_attn.q_proj.weight")
        k_proj = state_dict.pop(prefix + "self_attn.k_proj.weight")
        v_proj = state_dict.pop(prefix + "self_attn.v_proj.weight")
        return torch.cat(
            [
                q_proj.reshape((ng, dim * nh // ng, -1)),
                k_proj.reshape((ng, dim, -1)),
                v_proj.reshape((ng, dim, -1)),
            ],
            dim=1,
        ).reshape((-1, margs.hidden_size))
    else:
        raise ValueError(f"Invalid key ({key}) for llama model")

def olmo_get_tensor(state_dict, key, margs, layer=None):
    prefix = ""
    if layer is not None:
        prefix = f"model.transformer.blocks.{layer}."
    mapper = {
        "word embeddings": "model.transformer.wte.weight",
        "dense weight": "attn_out.weight",
        "mlp l1 weight": "ff_out.weight",
        "output layer weight": "model.transformer.ff_out.weight",
    }
    if key in mapper:
        return state_dict.pop(prefix + mapper[key])
    elif key == "mlp l0 weight W":
        return state_dict[prefix + "ff_proj.weight"].chunk(2, dim=0)[1]
    elif key == "mlp l0 weight V":
        return state_dict.pop(prefix + "ff_proj.weight").chunk(2, dim=0)[0]
    elif key == "qkv weight":
        q_proj, k_proj, v_proj = torch.chunk(state_dict.pop(prefix + "att_proj.weight"), 3, dim=0)
        ng = margs.num_attention_heads # TODO: group query attention in olmo models
        dim = margs.hidden_size
        nh = margs.num_attention_heads
        return torch.cat(
            [
                q_proj.reshape((ng, dim * nh // ng, -1)),
                k_proj.reshape((ng, dim, -1)),
                v_proj.reshape((ng, dim, -1)),
            ],
            dim=1,
        ).reshape((-1, margs.hidden_size))
    else:
        raise ValueError(f"Invalid key ({key}) for olmo model")

TENSOR_GETTERS = {
    "pythia": pythia_get_tensor,
    "llama": llama_get_tensor,
    "olmo": olmo_get_tensor,
}

def _load_checkpoint(queue, args):

    # Search in directory above this.
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
    )
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print(
            "Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting."
        )
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = [
        'script.py',
        '--no-masked-softmax-fusion',
        '--no-bias-act-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size',
        '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--tokenizer-type',
        'HFTokenizer',
        '--use-mcore-models',
        '--load',
        args.load_dir,
    ]

    margs = parse_args()
    if args in ARG_SETTERS:
        ARG_SETTERS[args.hf_model_type](args.repo, args.verision, args.cache_dir, args.hf_model_type, set_args=margs)
    else:
        print(f"Unknown model type {args.hf_model_type}. Exiting.")
        queue.put("exit")
        exit(1)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    # TODO: is this necessary?
    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('glu', False)

    # Determine how to make our models.
    assert args.model_type == 'GPT', 'pythia a GPT model.'
    margs.model_type = ModelType.encoder_or_decoder

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = ModelDescriptor(
        model_type=args.model_type,
        num_layers=margs.num_layers,
        hidden_size=margs.hidden_size,
        seq_length=margs.seq_length,
        num_attention_heads=margs.num_attention_heads,
        max_position_embeddings=margs.max_position_embeddings,
        tokenizer_type=margs.tokenizer_type,
        iteration=margs.iteration,
        params_dtype=margs.params_dtype,
        bert_binary_head=margs.bert_binary_head,
        output_layer=margs.untie_embeddings_and_output_weights,
        position_embedding_type=margs.position_embedding_type,
        linear_bias=margs.add_bias_linear,
        qkv_bias=margs.add_bias_linear or margs.add_qkv_bias,
        norm_has_bias=margs.normalization == "LayerNorm",
        norm_has_weight= margs.normalization != "NonParametricLayerNorm",
        glu=margs.glu,
        previous_tensor_parallel_size=tp_size,
        previous_pipeline_parallel_size=pp_size,
        true_vocab_size=margs.vocab_size,  # Indicates skipping padding in saver
        make_vocab_size_divisible_by=None,
        checkpoint_args=vars(
            margs
        ),  # Assuming margs can be directly converted or you adjust as needed
        consumed_train_samples=0,
        consumed_valid_samples=0,
    )

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    state_dict = load_state_dict(args)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    tensor_getter = TENSOR_GETTERS[args.hf_model_type]

    def put_tensor(message, key, layer=None):
        message[key] = tensor_getter(state_dict, key, margs, layer)

    # Send embeddings.
    message = {}
    put_tensor(message, "word embeddings")
    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        if md.norm_has_weight:
            put_tensor(message, "input norm weight", layer_num)
            put_tensor(message, "post norm weight", layer_num)
        if md.norm_has_bias:
            put_tensor(message, "input norm bias", layer_num)
            put_tensor(message, "post norm bias", layer_num)

        if md.glu:
            put_tensor(message, "mlp l0 weight W", layer_num)
            put_tensor(message, "mlp l0 weight V", layer_num)
        else:
            put_tensor(message, "mlp l0 weight", layer_num)

        put_tensor(message, "mlp l1 weight", layer_num)
        if md.linear_bias:
            put_tensor(message, "mlp l0 bias", layer_num)
            put_tensor(message, "mlp l1 bias", layer_num)

        put_tensor(message, "qkv weight", layer_num)
        put_tensor(message, "dense weight", layer_num)
        if md.qkv_bias:
            put_tensor(message, "qkv bias", layer_num)
            put_tensor(message, "dense bias", layer_num)

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {}
    if md.norm_has_weight:
        put_tensor(message, "final norm weight")
    if md.norm_has_bias:
        put_tensor(message, "final norm bias")
    queue_put("final norm", message)

    message = {}
    if md.output_layer:
        put_tensor(message, "output layer weight")
        queue_put("output layer", message)

    for k in state_dict:
        print(f"WARN: unprocessed key: {k}")

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
