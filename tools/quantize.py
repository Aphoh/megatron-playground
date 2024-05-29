from awq import AutoAWQForCausalLM
from awq.models.llama import SplitConfig
from filelock import FileLock
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
import json
from pathlib import Path
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import argparse
import subprocess
import torch
from torch import nn

cached_dataset = None
def get_cached_dataset():
    global cached_dataset
    if cached_dataset is None:
        cached_dataset = load_dataset("mli-will/slimpj-val", split="validation")
        print("Loaded dataset with size", len(cached_dataset))
    return cached_dataset

def get_data_subset(split: str, tokenizer):
    if split == "calib":
        data = get_cached_dataset().select(range(1000))
    elif split == "valid":
        data = get_cached_dataset().select(range(1000, 2000))
    text = tokenizer.eos_token.join(data["text"])
    text_bytes = len(text.encode("utf-8"))
    data = tokenizer(text, return_tensors="pt").input_ids
    del text
    text_tokens = data.numel()
    print(f"Split {split} has {text_tokens / 1e6:.2f}m tokens")
    return data, text_tokens / text_bytes

# do this in a separate method so the closure copies layer_idx
def make_act_stats_hook(layer_idx, act_stats, act_fn):
    def hook(module, input, output):
        per_channel = output.reshape(-1, output.shape[-1])
        act_stats["act_pct0"][layer_idx] += (per_channel > 0).sum(dim=0).to(torch.int32)
        act_stats["act_magn"][layer_idx] += act_fn(per_channel).abs().sum(dim=0)
    return hook

def get_act_stats(model: LlamaForCausalLM, tokenizer):
    cfg: LlamaConfig = model.config
    n_layers, n_intermediate = cfg.num_hidden_layers, cfg.intermediate_size
    act_stats = {
        "act_pct0": torch.zeros((n_layers, n_intermediate), dtype=torch.int32, device=model.device),
        "act_magn": torch.zeros((n_layers, n_intermediate), dtype=torch.float32, device=model.device),
    }
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hk = make_act_stats_hook(i, act_stats, layer.mlp.act_fn)
        hooks.append(layer.mlp.gate_proj.register_forward_hook(hk))
    data, _tokens_per_byte = get_data_subset("calib", tokenizer)
    _nll, n_tokens = eval_on_dataset(model, data) 
    for k, v in list(act_stats.items()):
        act_stats[k] = v.cpu().float() / n_tokens
    for hook in hooks:
        hook.remove()

    # Measure vector magnitudes and vector magnitudes under the LM head
    act_stats["vec_magn"] = torch.zeros((n_layers, n_intermediate), dtype=torch.float32, device=model.device)
    act_stats["lmvec_magn"] = torch.zeros((n_layers, n_intermediate), dtype=torch.float32, device=model.device)
    lm_head_weight = model.lm_head.weight
    for i, layer in tqdm(enumerate(model.model.layers), desc="Measuring vector magnitudes"):
        down_weight: torch.Tensor = layer.mlp.down_proj.weight.clone().detach()
        act_stats["vec_magn"][i] = torch.linalg.vector_norm(down_weight, ord=2, dim=0)
        down_weight_norm = model.model.norm(down_weight.T)
        head_weighed = F.linear(down_weight_norm.T, lm_head_weight)
        act_stats["lmvec_magn"][i] = torch.linalg.vector_norm(head_weighed, ord=2, dim=-1)

    act_stats["act_vec_magn"] = act_stats["vec_magn"] * act_stats["act_magn"]
    act_stats["act_lmvec_magn"] = act_stats["lmvec_magn"] * act_stats["act_magn"]

    for key in ["act_vec_magn", "act_lmvec_magn"]:
        maxv = act_stats[key].max()
        minv = act_stats[key].min()
        act_stats[key] = (act_stats[key] - minv) / (maxv - minv).cpu() # turn into 0-1 range

    return act_stats

def eval_on_dataset(model, data, batch_size=8):
    seqlen = model.config.max_position_embeddings
    model = model.eval()
    n_samples = data.numel() // (seqlen * batch_size)
    n_tokens = n_samples * seqlen * batch_size
    data = data[0, :n_tokens].view(n_samples, batch_size, seqlen)

    nlls = []
    with torch.inference_mode():
        with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
            for i in progress_bar:
                batch = data[i].to(model.device)
                logits = model(batch).logits
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = batch[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                nlls.append(loss)

                progress_bar.set_description(f"Perplexity {torch.stack(nlls).mean().exp():.3f}")
    nll = torch.stack(nlls).mean()
    return nll, n_tokens

def get_valid_eval_dict(model, tokenizer):
    dset, tokens_per_byte = get_data_subset("valid", tokenizer)
    nll, _n_tokens = eval_on_dataset(model, dset)
    return {"eval-ppl": torch.exp(nll).item(), "eval-bpb": nll.item() * tokens_per_byte}

def maybe_convert_to_hf(model_path: Path, tokenizer: str) -> Path:
    assert model_path.exists(), f"Model path {model_path} does not exist"
    if (model_path / "latest_checkpointed_iteration.txt").exists():
        output_path = model_path.parent / f"{model_path.name}-hf"
        if (output_path / "model.safetensors").exists():
            return output_path
        subprocess.check_call(["python", "tools/make_hf_model.py"] 
            + ["--model", str(model_path)]
            + ["--tokenizer", tokenizer]
            + ["--output", str(output_path)]
        )
        model_path = output_path

    assert (model_path / "model.safetensors").exists(), f"Model {model_path} is not a hf model"
    return model_path

def write_result(args, model_path: Path, model_cfg: LlamaConfig, results: dict):
    out_file = Path(args.out_file)
    with FileLock(out_file.with_suffix(".lock")):
        with out_file.open("a") as f:
            line = json.dumps({
                "model": model_path.name,
                "model_cfg": model_cfg.to_dict(),
                "args": vars(args),
                **results
            })
            f.write(line + "\n")

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
        write_result(args, model_path, base_model.config, res)
        del base_model


    res = {}
    if args.quant_load is not None:
        quant_path = Path(args.quant_load)
        assert quant_path.exists(), f"Quantized model {args.quant_load} does not exist"
        model = AutoAWQForCausalLM.from_quantized(str(quant_path), "model.safetensors", fuse_layers=False)
        res["quant_counts"] = [model.config.intermediate_size for i in range(model.config.num_hidden_layers)]
        res["non_q_counts"] = [0 for i in range(model.config.num_hidden_layers)]
    else:
        # Load model
        print("Quantizing")
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, "model.safetensors", 
        )
        if args.split_type is not None:
            split_cfg = SplitConfig(
                random_cols = args.split_type == "rand",
                top_cols = args.split_type == "top",
                bottom_cols = args.split_type == "bottom",
                n_split_constant = args.split_constant,
                n_split_top_thresh = args.split_top,
                n_split_bottom_thresh = args.split_bottom,
            )
            acts_path = Path(model_path) / "act_stats.pt"
            acts = None
            if acts_path.exists():
                print("Loaded activation stats from", acts_path)
                acts = torch.load(acts_path)
            else:
                print("Getting activation stats")
                acts = get_act_stats(model.to("cuda"), tokenizer)
                torch.save(acts, str(acts_path))
            split_metric = "act_" + args.split_metric
            assert split_metric in acts, f"Split metric {split_metric} not found in {acts.keys()}"
            res["quant_counts"], res["non_q_counts"] = model.split_mlp(split_cfg, split_metric=acts[split_metric])

        if model.config.intermediate_size % args.group_size != 0:
            args.group_size //= 2
            print(f"Group size is not a factor of hidden size. Using group size {args.group_size}")
            assert model.config.hidden_size % args.group_size == 0
        # Quantize
        quant_config = { "zero_point": True, "q_group_size": args.group_size, "w_bit": 4, "version": "GEMM" }
        model.quantize(tokenizer, quant_config=quant_config, calib_data="mli-will/slimpj-val", split="validation")

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
    model = model.to(device) # sometimes this gets loaded on CPU
    res |= get_valid_eval_dict(model, tokenizer)
    write_result(args, model_path, model.config, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument("--model", type=str, help="Path to the base model", required=True)
    parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer", default="EleutherAI/pythia-70m")
    parser.add_argument("--group-size", type=int, help="Group size for quantization", default=128)
    parser.add_argument("--quant-load", type=str, help="Directory to load the quantized model", default=None)
    parser.add_argument("--quant-save", type=str, help="Directory to save the quantized model", default=None)
    parser.add_argument("--out-file", type=str, help="Output file to save results", default="results.jsonl")
    parser.add_argument("--eval-base", action="store_true", help="Evaluate base model")

    parser.add_argument("--split-type", type=str, choices=["rand", "top", "bottom"], default=None)
    parser.add_argument("--split-metric", type=str, choices=["pct0", "magn", "vec_magn", "lmvec_magn"], default="pct0")
    parser.add_argument("--split-constant", type=int, default=None)
    parser.add_argument("--split-top", type=float, default=None)
    parser.add_argument("--split-bottom", type=float, default=None)

    args = parser.parse_args()
    assert not args.quant_save and args.split_type, "Can't save split model yet"
    main(args)
