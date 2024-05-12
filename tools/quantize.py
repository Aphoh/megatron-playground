from awq import AutoAWQForCausalLM
from awq.evaluation import eval_utils
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import json
import lm_eval
from pathlib import Path
import argparse
import subprocess
import torch

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
    # Evaluating base model
    print("Evaluating base model")

    if args.eval_base:
        base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device).to(torch.bfloat16)
        ppl = eval_utils.evaluate_perplexity(base_model, tokenizer)
        res = {"wikitext": ppl}
        #res = lm_eval.simple_evaluate(model=base_model, tasks=args.tasks, device=device)
        write_result(args.out_file, model_path, base_model.config, res)
        del base_model


    quant_path = model_path / args.quantized_subdir
    quant_config = { "zero_point": True, "q_group_size": args.group_size, "w_bit": 4, "version": "GEMM" }

    if not quant_path.exists():
        # Load model
        print("Quantizing")
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, "model.safetensors", fuse_layers=False, 
        )
        # Quantize
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
    ppl = eval_utils.evaluate_perplexity(model, tokenizer)
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