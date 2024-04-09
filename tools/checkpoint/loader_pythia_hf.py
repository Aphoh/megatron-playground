# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
from arg_utils import ModelDescriptor
from transformers import GPTNeoXConfig
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

    group.add_argument("--hf-cache-dir", type=str, default=None, help="Cache directory for Huggingface models.")
    group.add_argument("--repo", type=str, required=True, help="Huggingface model repository.")
    group.add_argument("--revision", type=str, default="main", help="Huggingface model revision.")
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of the megatron repository')

def load_args_from_checkpoint(args, margs):

    # Read Llama args.
    config = GPTNeoXConfig.from_pretrained(args.repo, revision=args.revision)
    tokenizer_file = hf_hub_download(args.repo, "tokenizer.json", revision=args.revision, cache_dir=args.hf_cache_dir)

    # Update Megatron args.
    margs.seq_length = config.max_position_embeddings
    margs.max_position_embeddings = margs.seq_length
    margs.hidden_size = config.hidden_size
    margs.num_attention_heads = config.num_attention_heads
    margs.num_layers = config.num_hidden_layers
    margs.global_batch_size = 1024
    margs.norm_epsilon = config.layer_norm_eps
    margs.iteration = 1 # '0', 'release' don't work
    margs.position_embedding_type = "rope"
    margs.rotary_percent=config.rotary_pct
    margs.use_parallel_residual = config.use_parallel_residual
    margs.tokenizer_type = "HFTokenizer"
    margs.vocab_file = tokenizer_file
    margs.fp16 = config.torch_dtype == "float16"
    margs.bf16 = config.torch_dtype == "bfloat16"
    margs.normalization = "LayerNorm"
    margs.untie_embeddings_and_output_weights = not config.tie_word_embeddings
    margs.vocab_size = config.vocab_size
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
    '''Set model params.'''

    # Load Huggingface model.
    weight_files = []
    if file_exists(args.repo, "pytorch_model.bin.index.json", revision=args.revision):
        index = hf_hub_download(args.repo, "pytorch_model.bin.index.json", revision=args.revision, cache_dir=args.hf_cache_dir)
        index_contents = json.load(open(index))
        weight_files.extend(list(set(index_contents["weight_map"].values())))
    else:
        weight_files.append("pytorch_model.bin")
    downloaded_weights = []
    for weight_file in weight_files:
        res = hf_hub_download(args.repo, weight_file, revision=args.revision, cache_dir=args.hf_cache_dir)
        downloaded_weights.append(res)

    state_dict = {}
    for weight_file in downloaded_weights:
        state_dict |= torch.load(weight_file, map_location="cpu")

    return state_dict


def _load_checkpoint(queue, args):

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_args, set_global_variables
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--tokenizer-type', 'HFTokenizer',
                '--use-mcore-models',
                '--load', args.load_dir
                ]

    margs = parse_args()
    load_args_from_checkpoint(args, margs)

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
        qkv_bias=margs.add_bias_linear or margs.margs.add_qkv_bias,
        norm_has_bias=margs.normalization == "LayerNorm",
        swiglu=margs.swiglu,
        previous_tensor_parallel_size=tp_size,
        previous_pipeline_parallel_size=pp_size,
        true_vocab_size=margs.vocab_size,  # Indicates skipping padding in saver
        make_vocab_size_divisible_by=None,
        checkpoint_args=vars(margs),  # Assuming margs can be directly converted or you adjust as needed
        consumed_train_samples=0,
        consumed_valid_samples=0
    )

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    state_dict = load_pythia_state_dict(args)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": state_dict.pop("gpt_neox.embed_in.weight")
    }
    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        prefix = f"gpt_neox.layers.{layer_num}."
        message["input norm weight"] =  prefix + 'input_layernorm.weight'
        message["post norm weight"] = prefix + 'post_attention_layernorm.weight'
        if md.norm_has_bias:
            message["input norm bias"] = prefix + 'input_layernorm.bias'
            message["post norm bias"] = prefix + 'post_attention_layernorm.bias'

        assert not md.swiglu, "For now, no swiglu :)"
        message["mlp l0 weight"] = prefix + 'mlp.dense_h_to_4h.weight'
        message["mlp l1 weight"] = prefix + 'mlp.dense_4h_to_h.weight'
        if md.bias_linear:
            message["mlp l0 bias"] = prefix + 'mlp.dense_h_to_4h.bias'
            message["mlp l1 bias"] = prefix + 'mlp.dense_4h_to_h.bias'

        message["qkv weight"] = prefix + 'attention.query_key_value.weight'
        message["dense weight"] = prefix + 'attention.dense.weight'
        if md.qkv_bias:
            message["qkv bias"] = prefix + 'attention.query_key_value.bias'
            message["dense bias"] = prefix + 'attention.dense.bias'
        
        for m in message:
            message[m] = state_dict.pop(message[m])

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        "weight": state_dict.pop("gpt_neox.final_layer_norm.weight"),
    }
    if md.norm_has_bias:
        message["bias"] = state_dict.pop("gpt_neox.final_layer_norm.bias")
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": state_dict.pop("embed_out.weight")
        }
        queue_put("output layer", message)

    for k in state_dict:
        print(f"Unprocessed key: {k}")

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
