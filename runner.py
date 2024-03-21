import sys
import os
import subprocess
import argparse
from dataclasses import dataclass
from typing import Optional, List
from transformers import GPTNeoXConfig
from huggingface_hub import hf_hub_download
from filelock import FileLock
import socket
from pathlib import Path
import torch
import math


@dataclass
class Arguments:
    name: str
    load_pythia: bool = False
    pythia_version: Optional[str] = None
    data_dir: str = "/data"
    ckpt_dir: str = "/checkpoint"
    hf_cache_dir: str = "/hf_cache"
    learning_rate: float = 6e-4
    tensorboard_dir: str = "/tensorboard"
    wandb_project: str = "megatron-dsparse"


rank = int(os.environ.get("SLURM_PROCID", "0"))
is_slurm: bool = os.environ.get("SLURM_PROCID") is not None


def print_rank_0(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description='Run a variety of different training tasks')
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument('--load-pythia', type=bool, default=False, help='Whether to load pythia')
    parser.add_argument(
        '--pythia-version', type=str, default=None, help='Pythia model version to load'
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
    parser.add_argument(
        "--tensorboard-dir", type=str, default="/tensorboard", help="Tensorboard dir"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="megatron-dsparse", help="Wandb project"
    )
    args = parser.parse_args()
    return Arguments(**vars(args))


def get_model_size(train_args: dict) -> int:
    emb_size = train_args["hidden_size"]
    n_layers = train_args["num_layers"]
    ffn_size = train_args["ffn_hidden_size"]
    vocab_size = 50304  # TODO: should I load this from the tokenizer?
    model_size = emb_size * vocab_size  # embeddings
    model_size += n_layers * emb_size * 4 * emb_size  # qkv and output projection
    model_size += n_layers * 2 * emb_size * ffn_size  # ffn
    if "untie_embeddings_and_output_weights" in train_args:
        model_size += emb_size * vocab_size  # output embeddings
    return model_size


def get_memory_usage(model_size: int) -> int:
    # 4 bytes per weight + 2 bytes per activation + 2 bytes per gradient + 2 bytes safety margin
    return model_size * (4 + 2 + 2 + 2)


def download_pythia(args) -> Path:
    pythia_ckpt_dir = Path(args.ckpt_dir) / "pythia" / args.pythia_version
    lock_file = Path(args.ckpt_dir) / "pythia.lock"
    with FileLock(lock_file):
        if not pythia_ckpt_dir.exists():
            # We're the first to get here, download the model
            print(f"Downloading pythia model on rank {rank}")
            model_bin_path = hf_hub_download(
                pythia_repo(args), 'pytorch_model.bin', cache_dir=args.hf_cache_dir
            )
            print(f"Model downloaded to {model_bin_path}")
            print(f"Converting model at {pythia_ckpt_dir}")
            res = subprocess.run(
                [
                    "python",
                    "tools/checkpoint/convert_pythia_ckpt.py",
                    model_bin_path,
                    pythia_ckpt_dir,
                ].split()
            )
            if res.returncode != 0:
                subprocess.run(["rm", "-r", pythia_ckpt_dir])
                subprocess.run(["scancel", os.environ.get("SLURM_JOB_ID")])
                raise RuntimeError("Failed to convert pythia checkpoint")

            print(f"Pythia checkpoint successfully converted to {pythia_ckpt_dir}")
    return pythia_ckpt_dir


def pythia_repo(args: Arguments) -> str:
    return f"EleutherAI/pythia-{args.pythia_version}"


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
    args = {}
    if args.load_pythia:
        repo = pythia_repo(args)
        print_rank_0(f"Downloading pythia checkpoint from {repo}")
        ckpt_loc = download_pythia(args)
        args["load"] = ckpt_loc
    return args


def get_model_arch_arguments(args: Arguments) -> dict:
    args = {"use_mcore_models": ()}
    if args.load_pythia:
        assert (
            args.pythia_version is not None
        ), 'Pythia version must be specified when loading pythia'
        repo = f'EleutherAI/pythia-{args.pythia_version}'
        print_rank_0(f"Loading Pythia config from {repo}")
        pythia_config = GPTNeoXConfig.from_pretrained(
            f'ElutherAI/pythia-{args.pythia_version}', cache_dir=args.hf_cache_dir
        )
        # Arguments from the config
        args["hidden_size"] = pythia_config.hidden_size
        args["init_method_std"] = pythia_config.initializer_range
        args["ffn_hidden_size"] = pythia_config.intermediate_size
        args["norm_epsilon"] = pythia_config.layer_norm_eps
        args["max_position_embeddings"] = pythia_config.max_position_embeddings
        args["num_attention_heads"] = pythia_config.num_attention_heads
        args["num_layers"] = pythia_config.num_hidden_layers
        args["rotary_percent"] = pythia_config.rotary_pct
        if not pythia_config.tie_word_embeddings:
            args["untie_embeddings_and_output_weights"] = ()

        args["use_parallel_residual"] = pythia_config.use_parallel_residual
        args["tokenizer"] = pythia_config.vocab_size

        # static arguments
        args["position_embedding_type"] = "rope"
        args["normalization"] = "LayerNorm"

        # download tokenizer
        tokenizer_path = hf_hub_download(repo, "tokenizer.json", cache_dir=args.hf_cache_dir)
        args["vocab_file"] = tokenizer_path
        args["tokenizer_type"] = "HFTokenizer"
        args["data_path"] = Path(args.data_dir) / "slimpj" / "slimpj-neox-c1c2_text_document"
        args["attention_dropout"] = 0.0
        args["hidden_dropout"] = 0.0
        args["weight_decay"] = 0.01
        args["seq_length"] = 2048

    return args


def get_training_arguments(args: Arguments) -> dict:
    args = {"learning_rate": args.learning_rate, "bf16": ()}
    if args.load_pythia:
        args["adam_beta1"] = 0.9
        args["adam_beta2"] = 0.95
        args["adam_epsilon"] = 1e-8
        args["global_batch_size"] = 1024

    return args


def get_logging_arguments(args: Arguments) -> dict:
    args = {
        "log_throughput": (),
        "log_timers_to_tensorboard": (),
        "log_validation_ppl_to_tensorboard": (),
        "log_interval": 10,
        "save_interval": 1000,
        "eval_interval": 100,
        "tensorboard_dir": args.tensorboard_dir,
        "wandb_project": args.wandb_project,
        "wandb_exp_name": args.name,
    }
    return args


def get_torchrun_args(args: Arguments) -> dict:
    args = {}
    if is_slurm:
        args["nnodes"] = int(os.environ["SLURM_STEP_NUM_NODES"])
        args["nproc_per_node"] = os.environ["SLURM_GPUS_ON_NODE"]
        if args["nnodes"] == 1:
            args["standalone"] = ()
        else:
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            hostnames = subprocess.run(
                f"scontrol show hostnames {nodelist}".split(), capture_output=True
            )
            if hostnames.returncode != 0:
                raise RuntimeError("Failed to get hostnames")
            head_node = hostnames.stdout.split()[0].decode("utf-8")
            try:
                # Resolve the head node's hostname to IP address
                head_node_ip = socket.gethostbyname(head_node)
                print(f"Head node IP: {head_node_ip}")
            except socket.gaierror:
                raise RuntimeError("Could not resolve the head node's IP address.")
            args["rdzv_id"] = os.environ["RANDOM"]
            args["rdzv_endpoint"] = f"{head_node_ip}:29500"
            args["rdzv_backend"] = "c10d"
    else:
        args["nnodes"] = 1
        args["standalone"] = ()
        args["nproc_per_node"] = torch.cuda.device_count()
    return args


def main():
    args = parse_args()

    train_args = (
        get_checkpoint_load_arguments(args)
        | get_model_arch_arguments(args)
        | get_training_arguments(args)
        | get_logging_arguments(args)
    )
    torchrun_args = get_torchrun_args(args)

    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if "micro_batch_size" not in train_args:
        model_size = get_model_size(train_args)
        memory_usage = get_memory_usage(model_size)
        num_gpus = torchrun_args["nnodes"] * torchrun_args["nproc_per_node"]
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        micro_batch_size = memory_usage // (num_gpus * gpu_memory)
        micro_batch_size = int(2 ** math.floor(math.log2(micro_batch_size)))
        print_rank_0(f"Expected memory usage (GB): {memory_usage/1e9:.2f}")
        print_rank_0(f"Setting micro batch size to: {micro_batch_size}")
        print_rank_0(f"Approximate memory usage per gpu (GB): {memory_usage / num_gpus / 1e9:.2f}")

    print_rank_0(f"Running with args: {train_args}")
    subprocess.run(
        ["torchrun"]
        + arg_dict_to_list(torchrun_args)
        + ["pretrain_gpt.py"]
        + arg_dict_to_list(train_args),
    )


if __name__ == "__main__":
    main()
