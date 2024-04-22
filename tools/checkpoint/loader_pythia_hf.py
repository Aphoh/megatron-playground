# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
from arg_utils import ModelDescriptor
import safetensors.torch as sttorch
from transformers import GPTNeoXConfig, GPTNeoXTokenizerFast, LlamaConfig, LlamaTokenizerFast
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError
from typing import Optional


def file_exists(
    repo_id: str,
    filename: str,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    url = hf_hub_url(repo_id=repo_id, repo_type=repo_type, revision=revision, filename=filename)
    try:
        get_hf_file_metadata(url, token=token)
        return True
    except (RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError):
        return False


def add_arguments(parser):
    group = parser.add_argument_group(title='Pythia HF loader')

    group.add_argument(
        "--hf-model-type",
        type=str,
        required=True,
        choices=["pythia", "llama"],
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


def load_llama_args(args, margs):

    config = LlamaConfig.from_pretrained(
        args.repo, revision=args.revision, cache_dir=args.hf_cache_dir
    )
    tokenizer_file = hf_hub_download(
        args.repo, "tokenizer.json", revision=args.revision, cache_dir=args.hf_cache_dir
    )
    tokenizer = LlamaTokenizerFast.from_pretrained(
        args.repo, revision=args.revision, cache_dir=args.hf_cache_dir
    )

    # Update Megatron args.
    margs.seq_length = config.max_position_embeddings
    margs.max_position_embeddings = config.max_position_embeddings
    margs.hidden_size = config.hidden_size
    margs.num_attention_heads = config.num_attention_heads
    margs.num_layers = config.num_hidden_layers
    margs.global_batch_size = 1024
    margs.norm_epsilon = config.rms_norm_eps
    margs.iteration = 1  # '0', 'release' don't work
    margs.position_embedding_type = "rope"
    margs.rotary_base = config.rope_theta
    margs.swiglu = True
    margs.tokenizer_type = "HFTokenizer"
    margs.vocab_file = tokenizer_file
    margs.bf16 = True
    margs.normalization = "RMSNorm"
    margs.add_bias_linear = False
    margs.untie_embeddings_and_output_weights = True
    margs.vocab_size = tokenizer.vocab_size
    margs.ffn_hidden_size = config.intermediate_size

    if hasattr(config, "num_key_value_heads"):
        margs.group_query_attention = True
        margs.num_query_groups = config.num_key_value_heads


def load_pythia_args(args, margs):

    # Read Llama args.
    config = GPTNeoXConfig.from_pretrained(args.repo, revision=args.revision)
    tokenizer_file = hf_hub_download(
        args.repo, "tokenizer.json", revision=args.revision, cache_dir=args.hf_cache_dir
    )
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(
        args.repo, revision=args.revision, cache_dir=args.hf_cache_dir
    )

    # Update Megatron args.
    margs.seq_length = config.max_position_embeddings
    margs.max_position_embeddings = margs.seq_length
    margs.hidden_size = config.hidden_size
    margs.num_attention_heads = config.num_attention_heads
    margs.num_layers = config.num_hidden_layers
    margs.global_batch_size = 1024
    margs.norm_epsilon = config.layer_norm_eps
    margs.iteration = 1  # '0', 'release' don't work
    margs.position_embedding_type = "rope"
    margs.rotary_percent = config.rotary_pct
    margs.use_parallel_residual = config.use_parallel_residual
    margs.tokenizer_type = "HFTokenizer"
    margs.vocab_file = tokenizer_file
    margs.fp16 = config.torch_dtype == "float16"
    margs.bf16 = config.torch_dtype == "bfloat16"
    margs.normalization = "LayerNorm"
    margs.untie_embeddings_and_output_weights = not config.tie_word_embeddings
    margs.vocab_size = tokenizer.vocab_size
    margs.ffn_hidden_size = config.intermediate_size
    margs.init_method_std = config.initializer_range
    if config.hidden_act == "relu":
        margs.relu = True
    else:
        assert config.hidden_act == "gelu"
    margs.hidden_dropout = config.hidden_dropout
    margs.attention_dropout = config.attention_dropout
    margs.weight_decay = 0.01
    assert config.attention_bias


def load_pythia_state_dict(args):
    # Load Huggingface model.
    weight_files = []
    if file_exists(args.repo, "pytorch_model.bin.index.json", revision=args.revision):
        index = hf_hub_download(
            args.repo,
            "pytorch_model.bin.index.json",
            revision=args.revision,
            cache_dir=args.hf_cache_dir,
        )
        index_contents = json.load(open(index))
        weight_files.extend(list(set(index_contents["weight_map"].values())))
    else:
        weight_files.append("pytorch_model.bin")
    downloaded_weights = []
    for weight_file in weight_files:
        res = hf_hub_download(
            args.repo, weight_file, revision=args.revision, cache_dir=args.hf_cache_dir
        )
        downloaded_weights.append(res)

    state_dict = {}
    for weight_file in downloaded_weights:
        state_dict |= torch.load(weight_file, map_location="cpu")

    return state_dict


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


def load_llama_state_dict(args):
    # load Huggingface model.
    weight_files = []
    if file_exists(args.repo, "model.safetensors.index.json", revision=args.revision):
        index = hf_hub_download(
            args.repo,
            "model.safetensors.index.json",
            revision=args.revision,
            cache_dir=args.hf_cache_dir,
        )
        index_contents = json.load(open(index))
        weight_files.extend(list(set(index_contents["weight_map"].values())))
    else:
        assert file_exists(args.repo, "model.safetensors")
        weight_files.append("model.safetensors")

    downloaded_weights = []
    for weight_file in weight_files:
        res = hf_hub_download(
            args.repo, weight_file, revision=args.revision, cache_dir=args.hf_cache_dir
        )
        downloaded_weights.append(res)

    state_dict = {}
    for weight_file in downloaded_weights:
        state_dict |= sttorch.load_file(weight_file, device="cpu")

    return state_dict


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
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron import fused_kernels
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
        '--no-bias-gelu-fusion',
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
    if args.hf_model_type == "pythia":
        load_pythia_args(args, margs)
    elif args.hf_model_type == "llama":
        load_llama_args(args, margs)
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
    check_for_arg('swiglu', False)

    # Determine how to make our models.
    assert args.model_type == 'GPT', 'pythia a GPT model.'
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

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
        bias_linear=margs.add_bias_linear,
        qkv_bias=margs.add_bias_linear or margs.add_qkv_bias,
        norm_has_bias=margs.normalization == "LayerNorm",
        swiglu=margs.swiglu,
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
    state_dict = {}
    if args.hf_model_type == "llama":
        state_dict = load_llama_state_dict(args)
    elif args.hf_model_type == "pythia":
        state_dict = load_pythia_state_dict(args)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    tensor_getter = pythia_get_tensor if args.hf_model_type == "pythia" else llama_get_tensor

    def put_tensor(message, key, layer=None):
        message[key] = tensor_getter(state_dict, key, margs, layer)

    # Send embeddings.
    message = {}
    put_tensor(message, "word embeddings")
    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        put_tensor(message, "input norm weight", layer_num)
        put_tensor(message, "post norm weight", layer_num)
        if md.norm_has_bias:
            put_tensor(message, "input norm bias", layer_num)
            put_tensor(message, "post norm bias", layer_num)

        if md.swiglu:
            put_tensor(message, "mlp l0 weight W", layer_num)
            put_tensor(message, "mlp l0 weight V", layer_num)
        else:
            put_tensor(message, "mlp l0 weight", layer_num)

        put_tensor(message, "mlp l1 weight", layer_num)
        if md.bias_linear:
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
