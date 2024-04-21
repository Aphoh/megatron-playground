import os
import subprocess
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple
from transformers import GPTNeoXConfig, LlamaConfig
from huggingface_hub import hf_hub_download
from filelock import FileLock
import socket
from pathlib import Path
import json
import torch
import math


@dataclass
class Arguments:
    name: str

    # torchrun arguments
    rank: int
    nnodes: int
    gpus_per_node: Optional[int] = None
    hostnames: Optional[List[str]] = None
    rdzv_id: Optional[str] = None

    # running args
    steps: int = 1000
    run_ldconfig: bool = False
    ## model loading
    load_repo: Optional[str] = None
    load_rev: str = "main"
    load_type: Optional[str] = None
    ## paths
    data_dir: str = "/data"
    checkpoint_dir: str = "/checkpoint"
    tensorboard_dir: str = "/tensorboard"
    hf_cache_dir: str = "/hf_cache"
    learning_rate: float = 6e-4
    lr_decay_time_fraction: float = 1.0
    lr_warmup_fraction: float = 0.01

    # logging
    wandb_project: str = "megatron-dsparse"

    # Should be bf16, fp16 or fp32
    dtype: str = "bf16"


def parse_args() -> Tuple[Arguments, list]:
    parser = argparse.ArgumentParser(
        description='Run a variety of different training tasks', allow_abbrev=False
    )
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--load-repo", type=str, default=None, help="Huggingface model to load")
    parser.add_argument("--load-rev", type=str, default="main", help="Huggingface revision to load")
    parser.add_argument(
        "--load-type",
        type=str,
        default=None,
        help="Huggingface model type to load",
        choices=["pythia", "llama", "llama3"],
    )
    parser.add_argument('--data-dir', type=str, default="/data", help='Location to load data')
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default="/checkpoint",
        help='Location to load/save checkpoints',
    )
    parser.add_argument(
        '--hf-cache-dir', type=str, default="/hf_cache", help='Location of huggingface cache'
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--lr-decay-time-fraction", type=float, default=1.0, help="Learning rate")
    parser.add_argument(
        "--tensorboard-dir", type=str, default="/tensorboard", help="Tensorboard dir"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="megatron-dsparse", help="Wandb project"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument(
        "--lr-warmup-fraction", type=float, default=0.01, help="Learning rate warmup fraction"
    )
    parser.add_argument("--run-ldconfig", action="store_true", help="Run ldconfig before training")

    parser.add_argument("--rank", type=int, help="Rank of the current process")
    parser.add_argument("--nnodes", type=int, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, help="Number of GPUs per node")
    parser.add_argument("--hostnames", type=str, help="Hostnames of the nodes involved")
    parser.add_argument("--rdzv-id", type=str, help="Rondevouz ID")

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type to use",
    )
    args, unknown = parser.parse_known_args()
    if args.rank is None:
        args.rank = int(os.environ["MGT_RANK"])
    if args.nnodes is None:
        args.nnodes = int(os.environ["MGT_WORLD_SIZE"])
    if args.nnodes > 1:
        if args.gpus_per_node is None:
            args.gpus_per_node = int(os.environ["MGT_NUM_GPUS"])
        if args.hostnames is None:
            args.hostnames = os.environ["MGT_HOSTS"]
        if args.rdzv_id is None:
            args.rdzv_id = os.environ["MGT_RZDV_ID"]

    if args.hostnames is not None:
        args.hostnames = [k for v in args.hostnames.split() for k in v.split(",")]
    if unknown and args.rank == 0:
        print(f"Got unknown arguments, passing to megatron: {unknown}")

    return Arguments(**vars(args)), unknown


def print_rank_0(args: Arguments, *varargs, **kwargs):
    if args.rank == 0:
        print(*varargs, **kwargs)

def arg_dict_to_list(args: dict) -> List[str]:
    res = []
    for k, v in args.items():
        k = "--" + k.replace("_", "-")
        if v == ():
            res.append(k)
        else:
            res.extend([k, str(v)])
    return res


def get_checkpoint_load_arguments(args: Arguments) -> dict:
    res = {"save": Path(args.checkpoint_dir) / args.name}
    if args.load_repo:
        model_name = args.load_repo.split("/")[-1].lower()
        model_loc = Path(args.checkpoint_dir) / "base-ckpts" / model_name
        load_loc = model_loc / args.load_rev
        lock_file = model_loc / "loader.lock"
        target_tp = 1
        if "13b" in args.load_repo:
            print(args, "Loading 13b model with TP=2")
            target_tp = 2
        if "70b" in args.load_repo:
            print(args, "Loading 70b model with TP=8")
            target_tp = 8
        res["tensor_model_parallel_size"] = target_tp
        with FileLock(lock_file):
            res["load"] = load_loc
            load_type = args.load_type
            if load_type == "llama3":
                load_type = "llama" # converter doesn't care about llama1/2 v.s. 3
            if not load_loc.exists():
                print("RANK", args.rank, "converting checkpoint for", args.load_repo)
                print_rank_0(args, f"Downloading checkpoint from {args.load_repo}")
                subprocess.run(
                    ["python", "tools/checkpoint/util.py"]
                    + ["--model-type", "GPT"]
                    + ["--loader", "pythia_hf"]
                    + ["--saver", "megatron"]
                    + ["--target-tensor-parallel-size", target_tp]
                    + ["--load-dir", "/tmp/ignore"]
                    + ["--save-dir", load_loc]
                    + ["--hf-cache-dir", args.hf_cache_dir]
                    + ["--repo", args.load_repo]
                    + ["--revision", args.load_rev],
                    + ["--hf-model-type", load_type],
                    check=True,
                )
                print(args, f"Successfully converted checkpoint to {load_loc}")
    return res

def get_pythia_args(args: Arguments) -> dict:
    res = {}
    print_rank_0(args, f"Loading Pythia config from {args.load_repo}")
    pythia_config = GPTNeoXConfig.from_pretrained(args.load_repo, revision=args.load_rev, cache_dir=args.hf_cache_dir)
    # Arguments from the config
    res["hidden_size"] = pythia_config.hidden_size
    res["init_method_std"] = pythia_config.initializer_range
    res["ffn_hidden_size"] = pythia_config.intermediate_size
    res["norm_epsilon"] = pythia_config.layer_norm_eps
    res["max_position_embeddings"] = pythia_config.max_position_embeddings
    res["num_attention_heads"] = pythia_config.num_attention_heads
    res["num_layers"] = pythia_config.num_hidden_layers
    res["rotary_percent"] = pythia_config.rotary_pct
    if not pythia_config.tie_word_embeddings:
        res["untie_embeddings_and_output_weights"] = ()

    if pythia_config.use_parallel_residual:
        res["use_parallel_residual"] = ()

    # static arguments
    res["position_embedding_type"] = "rope"
    res["normalization"] = "LayerNorm"

    # download tokenizer
    tokenizer_path = hf_hub_download(args.load_rev, "tokenizer.json", cache_dir=args.hf_cache_dir)
    res["vocab_file"] = tokenizer_path
    res["tokenizer_type"] = "HFTokenizer"
    res["data_path"] = Path(args.data_dir) / "slimpj" / "slimpj-neox-c1c2_text_document"
    res["attention_dropout"] = 0.0
    res["hidden_dropout"] = 0.0
    res["weight_decay"] = 0.01
    res["seq_length"] = 2048
    res["transformer_impl"] = "transformer_engine"

    return res

def get_llama_args(args: Arguments) -> dict:
    res = {}
    print_rank_0(args, f"Loading Llama config from {args.load_repo}")
    config = LlamaConfig.from_pretrained(args.load_repo, revision=args.load_rev, cache_dir=args.hf_cache_dir)
    tokenizer_file = hf_hub_download(args.load_rev, "tokenizer.json", cache_dir=args.hf_cache_dir)
    res["seq_length"] = config.max_position_embeddings
    res["max_position_embeddings"] = config.max_position_embeddings
    res["hidden_size"] = config.hidden_size
    res["num_attention_heads"] = config.num_attention_heads
    res["num_layers"] = config.num_hidden_layers
    res["global_batch_size"] = 1024
    res["norm_epsilon"] = config.rms_norm_eps
    res["iteration"] = 1  # '0', 'release' don't work
    res["position_embedding_type"] = "rope"
    res["rotary_base"] = config.rope_theta
    res["swiglu"] = True
    res["tokenizer_type"] = "HFTokenizer"
    res["vocab_file"] = tokenizer_file
    res["bf16"] = True
    res["normalization"] = "RMSNorm"
    res["add_bias_linear"] = False
    res["untie_embeddings_and_output_weights"] = True
    res["ffn_hidden_size"] = config.intermediate_size

    if hasattr(config, "num_key_value_heads"):
        res["group_query_attention"] = True
        res["num_query_groups"] = config.num_key_value_heads

    file = f"slimpj-{args.load_type}-c1_text_document"
    res["data_path"] = Path(args.data_dir) / "slimpj" / file
    res["attention_dropout"] = 0.0
    res["hidden_dropout"] = 0.0
    res["weight_decay"] = 0.01
    res["transformer_impl"] = "transformer_engine"

    return res

def get_model_arch_arguments(args: Arguments) -> dict:
    res = {"use_mcore_models": ()}
    if args.load_repo:
        if args.load_type == "pythia":
            res |= get_pythia_args(args)
        elif args.load_type == "llama" or args.load_type == "llama3":
            res |= get_llama_args(args)
    return res


def get_training_arguments(args: Arguments) -> dict:
    res = {"lr": args.learning_rate}
    if args.dtype == "bf16":
        res["bf16"] = ()
    elif args.dtype == "fp16":
        res["fp16"] = ()

    if args.load_pythia:
        res["adam_beta1"] = 0.9
        res["adam_beta2"] = 0.95
        res["adam_eps"] = 1e-8
        res["global_batch_size"] = 1024
        res["finetune"] = ()

    res["train_iters"] = args.steps
    res["lr_decay_iters"] = int(args.steps * args.lr_decay_time_fraction)
    res["lr_warmup_fraction"] = args.lr_warmup_fraction
    res["lr_decay_style"] = "cosine"
    res["min_lr"] = args.learning_rate * 0.1

    return res


def get_logging_arguments(args: Arguments) -> dict:
    res = {
        "log_throughput": (),
        "log_timers_to_tensorboard": (),
        "log_validation_ppl_to_tensorboard": (),
        "log_interval": 10,
        "save_interval": 1000,
        "eval_interval": 100,
        "tensorboard_dir": Path(args.tensorboard_dir) / args.name,
        "wandb_project": args.wandb_project,
        "wandb_exp_name": args.name,
    }
    return res


def calc_ideal_num_experts(dm1, dm2, c, H1, H2, nl1, nl2, v):
    # First term
    term1 = -(2 * dm2 * (2 * c + H2)) / c
    # Second term
    term2 = -v / nl2
    # Third term
    term3 = (dm1 * (2 * dm1 * (2 + H1) * nl1 + v)) / (dm2 * nl2)

    # Total quantity
    total_quantity = term1 + term2 + term3
    return total_quantity


def round_down_to_divisor(k, n):
    while k > 0:
        if n % k == 0:
            return k
        k -= 1
    return None  # In case no divisor is found which should not happen unless k <= 0


def get_torchrun_args(args: Arguments) -> dict:
    res = {}
    res["nnodes"] = args.nnodes
    if args.nnodes != 1:
        res["nproc_per_node"] = args.gpus_per_node
        head_node = args.hostnames[0]
        try:
            # Resolve the head node's hostname to IP address
            head_node_ip = socket.gethostbyname(head_node)
            print(f"Head node IP: {head_node_ip}")
        except socket.gaierror:
            raise RuntimeError("Could not resolve the head node's IP address.")
        res["rdzv_id"] = args.rdzv_id
        res["rdzv_endpoint"] = f"{head_node_ip}:29500"
        res["rdzv_backend"] = "c10d"
    else:
        res["standalone"] = ()
        res["nproc_per_node"] = args.gpus_per_node or torch.cuda.device_count()

    return res


def main():
    args, downstream_args = parse_args()

    if args.run_ldconfig:
        assert subprocess.run(["ldconfig"]).returncode == 0, "ldconfig failed"

    train_args = (
        get_checkpoint_load_arguments(args)
        | get_model_arch_arguments(args)
        | get_training_arguments(args)
        | get_logging_arguments(args)
    )
    torchrun_args = get_torchrun_args(args)

    # print environment variables
    res_args = (
        ["torchrun"]
        + arg_dict_to_list(torchrun_args)
        + ["pretrain_gpt.py"]
        + arg_dict_to_list(train_args)
        + downstream_args
    )
    print_rank_0(args, f"Running command: {' '.join(res_args)}", flush=True)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    subprocess.run(
        res_args,
        env=os.environ,
    )


if __name__ == "__main__":
    main()
