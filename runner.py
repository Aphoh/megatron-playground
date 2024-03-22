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
    load_pythia: Optional[str] = None
    data_dir: str = "/data"
    hf_cache_dir: str = "/hf_cache"
    learning_rate: float = 6e-4
    checkpoint_dir: str = "/checkpoint"
    tensorboard_dir: str = "/tensorboard"
    wandb_project: str = "megatron-dsparse"
    steps: int = 1000
    run_ldconfig: bool = False


rank = int(os.environ.get("SLURM_PROCID", "0"))
is_slurm: bool = os.environ.get("SLURM_PROCID") is not None


def print_rank_0(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description='Run a variety of different training tasks')
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument(
        '--load-pythia', type=str, default=None, help='Pythia model version to load'
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
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--run-ldconfig", action="store_true", help="Run ldconfig before training")
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
    fp32_size = model_size * 4  # store an fp32 copy of model weights in optimizer
    # , bytes per weight + 2 bytes per activation + 2 bytes per gradient + 2 bytes safety margin
    bf16_size = model_size * (2 + 2 + 2 + 2)
    return fp32_size + bf16_size


def download_pythia(args: Arguments) -> Path:
    pythia_ckpt_dir = Path(args.checkpoint_dir) / "pythia" / args.load_pythia
    lock_file = Path(args.checkpoint_dir) / "pythia.lock"
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
                ]
            )
            if res.returncode != 0:
                subprocess.run(["rm", "-r", pythia_ckpt_dir])
                subprocess.run(["scancel", os.environ.get("SLURM_JOB_ID")])
                raise RuntimeError("Failed to convert pythia checkpoint")

            print(f"Pythia checkpoint successfully converted to {pythia_ckpt_dir}")
    return pythia_ckpt_dir


def pythia_repo(args: Arguments) -> str:
    return f"EleutherAI/pythia-{args.load_pythia}"


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
    if args.load_pythia:
        repo = pythia_repo(args)
        print_rank_0(f"Downloading pythia checkpoint from {repo}")
        ckpt_loc = download_pythia(args)
        res["load"] = ckpt_loc
    return res


def get_model_arch_arguments(args: Arguments) -> dict:
    res = {"use_mcore_models": ()}
    if args.load_pythia:
        repo = pythia_repo(args)
        print_rank_0(f"Loading Pythia config from {repo}")
        pythia_config = GPTNeoXConfig.from_pretrained(repo, cache_dir=args.hf_cache_dir)
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
        tokenizer_path = hf_hub_download(repo, "tokenizer.json", cache_dir=args.hf_cache_dir)
        res["vocab_file"] = tokenizer_path
        res["tokenizer_type"] = "HFTokenizer"
        res["data_path"] = Path(args.data_dir) / "slimpj" / "slimpj-neox-c1c2_text_document"
        res["attention_dropout"] = 0.0
        res["hidden_dropout"] = 0.0
        res["weight_decay"] = 0.01
        res["seq_length"] = 2048

    return res


def get_training_arguments(args: Arguments) -> dict:
    res = {"lr": args.learning_rate, "bf16": ()}
    if args.load_pythia:
        res["adam_beta1"] = 0.9
        res["adam_beta2"] = 0.95
        res["adam_eps"] = 1e-8
        res["global_batch_size"] = 1024
        res["finetune"] = ()

    res["train_iters"] = args.steps
    res["lr_decay_iters"] = args.steps
    res["lr_warmup_fraction"] = 0.01
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


def get_torchrun_args(args: Arguments) -> dict:
    res = {}
    if is_slurm:
        res["nnodes"] = int(os.environ["SLURM_JOB_NUM_NODES"])
        res["nproc_per_node"] = int(os.environ["SLURM_GPUS_ON_NODE"])
        if res["nnodes"] == 1:
            res["standalone"] = ()
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
            res["rdzv_id"] = os.environ["RANDOM"]
            res["rdzv_endpoint"] = f"{head_node_ip}:29500"
            res["rdzv_backend"] = "c10d"
    else:
        res["nnodes"] = 1
        res["standalone"] = ()
        res["nproc_per_node"] = torch.cuda.device_count()
    return res


def main():
    args = parse_args()
    if args.run_ldconfig:
        assert subprocess.run(["ldconfig"]).returncode == 0, "ldconfig failed"

    train_args = (
        get_checkpoint_load_arguments(args)
        | get_model_arch_arguments(args)
        | get_training_arguments(args)
        | get_logging_arguments(args)
    )
    torchrun_args = get_torchrun_args(args)

    if "micro_batch_size" not in train_args:
        model_size = get_model_size(train_args)
        memory_usage = get_memory_usage(model_size)
        print_rank_0(f"Expected memory usage (GB): {memory_usage/1e9:.2f}")
        num_gpus = int(torchrun_args["nnodes"]) * int(torchrun_args["nproc_per_node"])
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        micro_batch_size = (num_gpus * gpu_memory) / memory_usage
        print_rank_0(f"Calculated micro batch size: {micro_batch_size:.2f}")
        micro_batch_size = int(2 ** math.floor(math.log2(micro_batch_size)))
        print_rank_0(f"Setting micro batch size to: {micro_batch_size}")
        print_rank_0(
            f"Approximate memory usage per gpu (GB): {micro_batch_size * memory_usage / num_gpus / 1e9:.2f}"
        )
        train_args["micro_batch_size"] = micro_batch_size

    # print environment variables
    print_rank_0(f"Running with args: {train_args}", flush=True)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    subprocess.run(
        ["torchrun"]
        + arg_dict_to_list(torchrun_args)
        + ["pretrain_gpt.py"]
        + arg_dict_to_list(train_args),
        env=os.environ,
    )


if __name__ == "__main__":
    main()
