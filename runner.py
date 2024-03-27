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
    load_pythia: Optional[str] = None
    load_checkpoint: Optional[str] = None
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

    # dsparse arguments
    do_dsparse: bool = False
    from_model_size: Optional[str] = None
    to_model_size: Optional[str] = None
    ## 1/dsparse_factor is the fraction of the experts each token is routed to
    dsparse_factor: Optional[int] = None
    dsparse_start_t: float = 1.0
    dsparse_lr_mult: float = 1.0


def model_config_from_size(model_size: str) -> dict:
    # Assumes ffn hidden size of 4*hidden_size
    # num_layers, hidden_size, heads
    sizes = {
        "14m": (6, 512, 8),
        "160m": (12, 768, 12),
        "410m": (24, 1024, 16),
        "1.0b": (16, 2048, 8),
        "1.4b": (24, 2048, 16),
        "2.8b": (32, 2560, 32),
        "6.9b": (32, 4096, 32),
        "12b": (36, 5120, 40),
    }
    num_layers, hidden_size, heads = sizes[model_size]
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": heads,
    }


def print_rank_0(args: Arguments, *varargs, **kwargs):
    if args.rank == 0:
        print(*varargs, **kwargs)


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

    parser.add_argument("--do-dsparse", action="store_true", help="Enable dsparse")
    parser.add_argument("--from-model-size", type=str, help="Starting model size")
    parser.add_argument("--to-model-size", type=str, help="Ending model size")
    parser.add_argument("--dsparse-factor", type=int, help="dsparse factor")
    parser.add_argument("--dsparse-start-t", type=float, default=1.0, help="dsparse start t")
    parser.add_argument("--dsparse-lr-mult", type=float, default=1.0, help="dsparse lr mult")
    args = parser.parse_args()

    if args.rank is None:
        args.rank = int(os.environ["MGT_RANK"])
    if args.nnodes is None:
        args.nnodes = int(os.environ["MGT_WORLD_SIZE"])
    if args.nnodes != 1 and args.gpus_per_node is None:
        args.gpus_per_node = int(os.environ["MGT_NUM_GPUS"])
    if args.nnodes != 1 and args.hostnames is None:
        args.hostnames = os.environ["MGT_HOSTS"]
    if args.hostnames is not None:
        args.hostnames = [k for v in args.hostnames.split() for k in v.split(",")]
    if args.rdzv_id is None:
        args.rdzv_id = os.environ["MGT_RZDV_ID"]
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
    if train_args.get("dsparse_factor", False):
        model_size += n_layers * emb_size * train_args["dsparse_nblocks"]
    return model_size


def get_memory_usage(train_args: dict) -> int:
    model_size = get_model_size(train_args)
    fp32_size = model_size * 4  # store an fp32 copy of model weights in optimizer
    # 2 bytes per weight + 2 bytes per activation + 2 bytes per gradient + 2 bytes safety margin
    bf16_size = model_size * (2 + 2 + 2 + 2)
    return fp32_size + bf16_size


def download_pythia(args: Arguments) -> Path:
    pythia_ckpt_dir = Path(args.checkpoint_dir) / "pythia" / args.load_pythia
    lock_file = Path(args.checkpoint_dir) / "pythia.lock"
    with FileLock(lock_file):
        if not pythia_ckpt_dir.exists():
            # We're the first to get here, download the model
            print(f"Downloading pythia model on rank {args.rank}")
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
        print_rank_0(args, f"Downloading pythia checkpoint from {repo}")
        ckpt_loc = download_pythia(args)
        res["load"] = ckpt_loc
    return res


def get_model_arch_arguments(args: Arguments) -> dict:
    res = {"use_mcore_models": ()}
    if args.load_pythia:
        repo = pythia_repo(args)
        print_rank_0(args, f"Loading Pythia config from {repo}")
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
        res["transformer_impl"] = "transformer_engine"

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


def get_dsparse_arguments(args: Arguments, so_far: dict) -> dict:
    res = {}
    if args.do_dsparse:
        assert args.dsparse_factor is not None, "dsparse factor must be set"
        res["dsparse_factor"] = args.dsparse_factor
        from_model_size = model_config_from_size(args.from_model_size)
        to_model_size = model_config_from_size(args.to_model_size)
        will_load_size = {
            "num_layers": so_far["num_layers"],
            "hidden_size": so_far["hidden_size"],
            "num_attention_heads": so_far["num_attention_heads"],
        }
        assert (
            from_model_size == will_load_size
        ), f"Model size mismatch: {from_model_size} != {will_load_size}"
        num_exp = calc_ideal_num_experts(
            to_model_size["hidden_size"],
            from_model_size["hidden_size"],
            args.dsparse_factor,
            4,  # TODO: accept this as a hp
            4,
            to_model_size["num_layers"],
            from_model_size["num_layers"],
            50304,
        )
        num_exp = int(round(num_exp))
        assert num_exp > 1, "Number of experts must be > 1"
        div_num_exp = round_down_to_divisor(num_exp, 4 * from_model_size["hidden_size"])
        assert (
            div_num_exp is not None
        ), f"num_exp={num_exp}, hidden_size={4*from_model_size['hidden_size']}"
        res["dsparse_nblocks"] = div_num_exp
        print_rank_0(
            args,
            f"For dsparse factor {args.dsparse_factor} and hidden size {from_model_size['hidden_size']}",
        )
        print_rank_0(
            args, f"ideal num experts={num_exp:.2f}, rounded down to divisor={div_num_exp}"
        )
        res["dsparse_start_t"] = args.dsparse_start_t
        res["dsparse_lr_mult"] = args.dsparse_lr_mult
        res["dsparse_normalize_mask"] = ()
        res["dsparse_finetune"] = ()
        res["dsparse_anneal"] = ()
        res["dsparse_router_init_method"] = "const"

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
    args = parse_args()
    if args.run_ldconfig:
        assert subprocess.run(["ldconfig"]).returncode == 0, "ldconfig failed"

    train_args = (
        get_checkpoint_load_arguments(args)
        | get_model_arch_arguments(args)
        | get_training_arguments(args)
        | get_logging_arguments(args)
    )
    train_args |= get_dsparse_arguments(args, train_args)
    torchrun_args = get_torchrun_args(args)

    if "micro_batch_size" not in train_args:
        memory_usage = get_memory_usage(train_args)
        print_rank_0(args, f"Expected memory usage (GB): {memory_usage/1e9:.2f}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        micro_batch_size = gpu_memory / memory_usage
        print_rank_0(args, f"Calculated micro batch size: {micro_batch_size:.2f}")
        micro_batch_size = int(2 ** math.floor(math.log2(micro_batch_size)))
        print_rank_0(args, f"Setting micro batch size to: {micro_batch_size}")
        print_rank_0(
            args,
            f"Approximate memory usage per gpu (GB): {micro_batch_size * memory_usage / 1e9:.2f}",
        )
        train_args["micro_batch_size"] = micro_batch_size

    # print environment variables
    print_rank_0(args, f"Running with args: {train_args}", flush=True)
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
