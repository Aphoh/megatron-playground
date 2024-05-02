import os
import subprocess
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple
from filelock import FileLock
from tools.checkpoint.arg_utils import ARG_SETTERS
from runner_utils import get_ccla_config
import socket
from pathlib import Path
import torch


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
    load_ccla_config: Optional[int] = None
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
    do_save: bool = True
    reload: bool = False


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
        choices=["pythia", "llama", "llama3", "olmo"],
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
    parser.add_argument(
        "--load-ccla-config",
        type=int,
        default=None,
        help="Load a ccla config by number of million params",
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
    parser.add_argument(
        "--no-save", action="store_false", dest="do_save", help="Don't save checkpoints"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Reload the model from the checkpoint"
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

    if args.load_ccla_config:
        assert not args.load_repo, "Cannot specify both load-repo and load-ccla-config"

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
    res = {"wandb_save_dir": Path(args.checkpoint_dir) / args.name / "wandb"}
    if args.do_save:
        res["save"] = Path(args.checkpoint_dir) / args.name
    if args.reload:
        res["load"] = Path(args.checkpoint_dir) / args.name
    elif args.load_repo:
        model_name = args.load_repo.split("/")[-1].lower()
        model_loc = Path(args.checkpoint_dir) / "base-ckpts" / model_name
        load_loc = model_loc / args.load_rev
        lock_file = model_loc / "loader.lock"
        target_tp = 1
        tp_map = {
            "13b": 2,
            "70b": 8,
            "8b": 2,
            "7b": 2,
            "6.9b": 2,
        }
        for k, v in tp_map.items():
            if k in model_name:
                target_tp = v
                break
        res["tensor_model_parallel_size"] = target_tp
        with FileLock(lock_file):
            res["load"] = load_loc
            load_type = args.load_type
            if load_type == "llama3":
                load_type = "llama"  # converter doesn't care about llama1/2 v.s. 3
            if not load_loc.exists():
                print("RANK", args.rank, "converting checkpoint for", args.load_repo)
                print_rank_0(args, f"Downloading checkpoint from {args.load_repo}")
                subprocess.run(
                    ["python", "tools/checkpoint/convert.py"]
                    + ["--model-type", "GPT"]
                    + ["--loader", "pythia_hf"]
                    + ["--saver", "megatron"]
                    + ["--target-tensor-parallel-size", str(target_tp)]
                    + ["--load-dir", "/tmp/ignore"]
                    + ["--save-dir", str(load_loc)]
                    + ["--hf-cache-dir", args.hf_cache_dir]
                    + ["--repo", args.load_repo]
                    + ["--revision", args.load_rev]
                    + ["--hf-model-type", load_type],
                    check=True,
                )
                print(args, f"Successfully converted checkpoint to {load_loc}")
    return res


def get_model_arch_arguments(args: Arguments) -> dict:
    res = {"use_mcore_models": ()}
    if args.load_repo:
        if args.load_type in ARG_SETTERS:
            res |= ARG_SETTERS[args.load_type](
                args.load_repo, args.load_rev, args.hf_cache_dir, args.data_dir, args.load_type
            )
        else:
            raise ValueError(f"Unknown load type {args.load_type}")
    elif args.load_ccla_config:
        res |= get_ccla_config(args.load_ccla_config, args.hf_cache_dir, args.data_dir)
    return res


def get_training_arguments(args: Arguments) -> dict:
    if args.load_ccla_config:
        lr = get_ccla_config(args.load_ccla_config, args.hf_cache_dir, args.data_dir)["lr"]
        if lr != args.learning_rate:
            print_rank_0(
                args, f"Overriding learning rate {args.learning_rate} with {lr} from ccla config"
            )
            args.learning_rate = lr
    res = {"lr": args.learning_rate}
    if args.dtype == "bf16":
        res["bf16"] = ()
    elif args.dtype == "fp16":
        res["fp16"] = ()

    if args.load_repo and not args.reload:
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
        "eval_interval": 100,
        "tensorboard_dir": Path(args.tensorboard_dir) / args.name,
        "wandb_project": args.wandb_project,
        "wandb_exp_name": args.name,
    }
    if args.do_save:
        res["save_interval"] = 1000
    return res


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
        res_args, env=os.environ,
    )


if __name__ == "__main__":
    main()
