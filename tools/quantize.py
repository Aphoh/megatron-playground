from awq import AutoAWQForCausalLM
from awq.models.llama import SplitConfig
from filelock import FileLock
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import json
from pathlib import Path
import argparse
import subprocess
import torch
import wandb
from quantize.act_stats import log_act_stats, get_act_stats
from quantize.eval import get_valid_eval_dict


def maybe_convert_to_hf(model_path: Path, tokenizer: str) -> Path:
    assert model_path.exists(), f"Model path {model_path} does not exist"
    if (model_path / "latest_checkpointed_iteration.txt").exists():
        output_path = model_path.parent / f"{model_path.name}-hf"
        if (output_path / "model.safetensors").exists():
            return output_path
        subprocess.check_call(
            ["python", "tools/make_hf_model.py"]
            + ["--model", str(model_path)]
            + ["--tokenizer", tokenizer]
            + ["--output", str(output_path)]
        )
        model_path = output_path

    assert (model_path / "model.safetensors").exists(), f"Model {model_path} is not a hf model"
    return model_path


def get_split_config(args, act_stats):
    res = {}
    if args.split_constant is not None:
        res["n_split_constant"] = args.split_constant
    elif args.split_top is not None:
        res["n_split_top_thresh"] = args.split_top
    elif args.split_bottom is not None:
        res["n_split_bottom_thresh"] = args.split_bottom
    else:
        sorted_vals, _ = act_stats["act_" + args.split_metric].flatten().sort(descending=True)
        if args.split_top_pct is not None:
            res["n_split_top_thresh"] = sorted_vals[int(len(sorted_vals) * args.split_top_pct)]
        elif args.split_bottom_pct is not None:
            res["n_split_bottom_thresh"] = sorted_vals[-int(len(sorted_vals) * args.split_bottom_pct)]
        else:
            raise ValueError("No split threshold specified")
    return SplitConfig(
        random_cols=args.split_type == "rand",
        top_cols=args.split_type == "top",
        bottom_cols=args.split_type == "bottom",
        **res,
    )

def write_result(args, model_path: Path, model_cfg: LlamaConfig, results: dict):
    out_file = Path(args.out_file)
    wandb.config.update({"model_cfg": model_cfg.to_dict()})
    for k, v in results.items():
        if isinstance(v, list):
            table = wandb.Table(data=[(i, v[i]) for i in range(len(v))],columns=["layer", k])
            wandb.log({f"{k}_per_layer": wandb.plot.line(table, "layer", k, title=f"{k} per layer")})
        elif isinstance(v, float) or isinstance(v, int):
            wandb.log({k: v})
        else:
            print(f"WANDB: Skipping logging {k} as it is not a scalar or list")
    with FileLock(out_file.with_suffix(".lock")):
        with out_file.open("a") as f:
            line = json.dumps(
                {
                    "model": model_path.name,
                    "model_cfg": model_cfg.to_dict(),
                    "args": vars(args),
                    **results,
                }
            )
            f.write(line + "\n")


@torch.no_grad()
@torch.inference_mode()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model)
    model_path = maybe_convert_to_hf(model_path, args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Got model")

    if args.eval_base:
        print("Loading base model")
        base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device).to(torch.bfloat16)
        print("Evaluating base model")
        res = get_valid_eval_dict(base_model, tokenizer)
        res["non_q_counts"] = [
            base_model.config.intermediate_size for i in range(base_model.config.num_hidden_layers)
        ]
        res["quant_counts"] = [0 for i in range(base_model.config.num_hidden_layers)]
        write_result(args, model_path, base_model.config, res)
        del base_model

    res = {}
    if args.quant_load is not None:
        quant_path = Path(args.quant_load)
        assert quant_path.exists(), f"Quantized model {args.quant_load} does not exist"
        model = AutoAWQForCausalLM.from_quantized(
            str(quant_path), "model.safetensors", fuse_layers=False
        )
        res["quant_counts"] = [
            model.config.intermediate_size for i in range(model.config.num_hidden_layers)
        ]
        res["non_q_counts"] = [0 for i in range(model.config.num_hidden_layers)]
    else:
        # Load model
        print("Quantizing")
        model = AutoAWQForCausalLM.from_pretrained(model_path, "model.safetensors")
        if args.split_type is not None:
            acts_path = Path(model_path) / "act_stats.pt"
            acts = None
            with FileLock(acts_path.with_suffix(".lock")):
                if acts_path.exists():
                    print("Loaded activation stats from", acts_path)
                    acts = torch.load(acts_path)
                else:
                    print("Getting activation stats")
                    acts = get_act_stats(model.to("cuda"), tokenizer)
                    torch.save(acts, str(acts_path))
            log_act_stats(acts)
            name = f"{model_path.stem}-act-stats"
            if not wandb.Api().artifact_exists(name):
                artifact = wandb.Artifact(name, type="act-stats")
                artifact.add_file(acts_path)
                wandb.log_artifact(artifact)
            split_metric = "act_" + args.split_metric
            assert split_metric in acts, f"Split metric {split_metric} not found in {acts.keys()}"
            split_cfg = get_split_config(args, acts)
            wandb.config.update({"split_cfg": vars(split_cfg)})
            res["quant_counts"], res["non_q_counts"] = model.split_mlp(
                split_cfg, split_metric=acts[split_metric]
            )

        if model.config.intermediate_size % args.group_size != 0:
            args.group_size //= 2
            print(f"Group size is not a factor of hidden size. Using group size {args.group_size}")
            assert model.config.hidden_size % args.group_size == 0
        # Quantize
        quant_config = {
            "zero_point": True,
            "q_group_size": args.group_size,
            "w_bit": 4,
            "version": "GEMM",
        }
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data="mli-will/slimpj-val",
            split="validation",
        )

        # Save quantized model
        if args.quant_save is not None:
            print("Saving quantized model")
            quant_path = Path(args.quant_save)
            model.save_quantized(str(quant_path))
            tokenizer.save_pretrained(str(quant_path))
            print(f'Model is quantized and saved at "{quant_path}"')
        model = model.model

    print("Evaluating quantized model")
    # Evaluate
    model = model.to(device)  # sometimes this gets loaded on CPU
    res |= get_valid_eval_dict(model, tokenizer)
    write_result(args, model_path, model.config, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model")
    
    parser.add_argument("--wandb-project", type=str, help="Wandb project name", default="megatron-quantize")
    parser.add_argument("--model", type=str, help="Path to the base model", required=True)
    parser.add_argument(
        "--tokenizer", type=str, help="Path to the tokenizer", default="EleutherAI/pythia-70m"
    )
    parser.add_argument("--group-size", type=int, help="Group size for quantization", default=128)
    parser.add_argument(
        "--quant-load", type=str, help="Directory to load the quantized model", default=None
    )
    parser.add_argument(
        "--quant-save", type=str, help="Directory to save the quantized model", default=None
    )
    parser.add_argument(
        "--out-file", type=str, help="Output file to save results", default="results.jsonl"
    )
    parser.add_argument("--eval-base", action="store_true", help="Evaluate base model")

    parser.add_argument("--split-type", type=str, choices=["rand", "top", "bottom"], default=None, help="""
        How to choose the columns to split. This is separate from how many columns to choose, and is useful for
        doing ablations (ex: swapping in random columns for the top columns when splitting the top k%).
    """)
    parser.add_argument(
        "--split-metric",
        type=str,
        choices=["pct0", "magn", "vec_magn", "lmvec_magn"],
        default="pct0",
    )
    parser.add_argument("--split-constant", type=int, default=None, help="Skip quantizing a constant number of columns per layer")
    parser.add_argument("--split-top", type=float, default=None, help="Skip quantizing the values of split-metric above this threshold")
    parser.add_argument("--split-bottom", type=float, default=None, help="Skip quantizing the values of split-metric below this threshold")
    parser.add_argument("--split-top-pct", type=float, default=None, help="Skip quantizing the top pct of values of split-metric")
    parser.add_argument("--split-bottom-pct", type=float, default=None, help="Skip quantizing the bottom pct of values of split-metric")

    args = parser.parse_args()
    assert not args.quant_save and args.split_type, "Can't save split model yet"
    
    wandb.init(project=args.wandb_project, config=vars(args))

    main(args)
