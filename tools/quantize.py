from awq import AutoAWQForCausalLM
from awq.evaluation import eval_utils
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import json
import lm_eval
from pathlib import Path
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import argparse
import subprocess
import torch
from torch import nn

def evaluate_perplexity(model, tokenizer):
    model = model.to("cuda")
    assert model.device.type == "cuda"

    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    data = data.select(range(1000, 2000))
    data = tokenizer(tokenizer.eos_token.join(data["text"]), return_tensors="pt")
    data = data.input_ids.to(model.device)
    print("Got data shape", data.shape)

    seqlen = 2048
    batch_size = 8
    model = model.eval()
    n_samples = data.numel() // (seqlen * batch_size)
    data = data[0, :n_samples * seqlen * batch_size].view(n_samples, batch_size, seqlen)

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            batch = data[i].to(model.device)
            with torch.inference_mode():
                logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()

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

def write_result(out_file: Path, model_path: Path, model_cfg: LlamaConfig, results: dict):
    with out_file.open("a") as f:
        line = json.dumps({
            "model": model_path.name,
            "model_cfg": model_cfg.to_dict(),
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
        ppl = evaluate_perplexity(base_model, tokenizer)
        res = {"wikitext": ppl}
        #res = lm_eval.simple_evaluate(model=base_model, tasks=args.tasks, device=device)
        write_result(args.out_file, model_path, base_model.config, res)
        del base_model


    quant_path = model_path / args.quantized_subdir

    if not quant_path.exists():
        # Load model
        print("Quantizing")
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, "model.safetensors", 
        )
        acts_path = Path(model_path) / "acts.npy"
        if acts_path.exists():
            acts = torch.from_numpy(np.load(str(acts_path)))
            model.split_mlp(acts, 0.99)

        #print("Pre quant post split eval")
        #evaluate_perplexity(model.model, tokenizer)
        if model.config.intermediate_size % args.group_size != 0:
            args.group_size //= 2
            print(f"Group size is not a factor of hidden size. Using group size {args.group_size}")
            assert model.config.hidden_size % args.group_size == 0
        # Quantize
        quant_config = { "zero_point": True, "q_group_size": args.group_size, "w_bit": 4, "version": "GEMM" }
        model.quantize(tokenizer, quant_config=quant_config)

        print("Saving quantized model")
        # Save quantized model
        model.save_quantized(str(quant_path))
        tokenizer.save_pretrained(str(quant_path))
        print(f'Model is quantized and saved at "{quant_path}"')
        model = model.model
    else:
        model = AutoAWQForCausalLM.from_quantized(str(quant_path), "model.safetensors", fuse_layers=False)

    print("Evaluating quantized model")
    # Evaluate
    model = model.to(device)
    ppl = evaluate_perplexity(model, tokenizer)
    res = {"wikitext": ppl}
    #res = lm_eval.simple_evaluate(model=model, tasks=args.tasks, out_file=args.out_file)
    write_result(args.out_file, quant_path, model.config, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument("--model", type=str, help="Path to the base model", required=True)
    parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer", default="EleutherAI/pythia-70m")
    parser.add_argument("--group-size", type=int, help="Group size for quantization", default=128)
    parser.add_argument("--quantized-subdir", type=str, help="Subdirectory to save quantized model", default="quant_v0")
    parser.add_argument("--tasks", type=str, nargs="+", help="Tasks to evaluate on", default=[])
    parser.add_argument("--out-file", type=str, help="Output file to save results", default="results.jsonl")
    parser.add_argument("--eval-base", action="store_true", help="Evaluate base model")
    args = parser.parse_args()
    args.out_file = Path(args.out_file)
    main(args)
