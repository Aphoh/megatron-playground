import torch
from tqdm import tqdm
from torch import nn
from datasets import load_dataset

cached_dataset = None


def get_cached_dataset():
    global cached_dataset
    if cached_dataset is None:
        cached_dataset = load_dataset("mli-will/slimpj-val", split="validation").shuffle(seed=42)
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
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)

                progress_bar.set_description(f"Perplexity {torch.stack(nlls).mean().exp():.3f}")
    nll = torch.stack(nlls).mean()
    return nll, n_tokens


def get_valid_eval_dict(model, tokenizer):
    dset, tokens_per_byte = get_data_subset("valid", tokenizer)
    nll, _n_tokens = eval_on_dataset(model, dset)
    return {"eval-ppl": torch.exp(nll).item(), "eval-bpb": nll.item() * tokens_per_byte}
