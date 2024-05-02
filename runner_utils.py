from huggingface_hub import hf_hub_download
from pathlib import Path

ccla_configs = [
    {
        'name': 74,
        'hidden_size': 640,
        'ffn_hidden_size': 2560,
        'num_attention_heads': 10,
        'num_layers': 10,
        'lr': 0.001,
    },
    {
        'name': 90,
        'hidden_size': 640,
        'ffn_hidden_size': 2560,
        'num_attention_heads': 10,
        'num_layers': 13,
        'lr': 0.0008,
    },
    {
        'name': 140,
        'hidden_size': 768,
        'ffn_hidden_size': 3072,
        'num_attention_heads': 12,
        'num_layers': 15,
        'lr': 0.0006,
    },
    {
        'name': 175,
        'hidden_size': 896,
        'ffn_hidden_size': 3584,
        'num_attention_heads': 14,
        'num_layers': 14,
        'lr': 0.0006,
    },
    {
        'name': 251,
        'hidden_size': 1024,
        'ffn_hidden_size': 4096,
        'num_attention_heads': 16,
        'num_layers': 16,
        'lr': 0.0005,
    },
    {
        'name': 425,
        'hidden_size': 1280,
        'ffn_hidden_size': 5120,
        'num_attention_heads': 10,
        'num_layers': 18,
        'lr': 0.0003,
    },
    {
        'name': 587,
        'hidden_size': 1408,
        'ffn_hidden_size': 5632,
        'num_attention_heads': 11,
        'num_layers': 21,
        'lr': 0.0003,
    },
    {
        'name': 1018,
        'hidden_size': 1792,
        'ffn_hidden_size': 7168,
        'num_attention_heads': 14,
        'num_layers': 25,
        'lr': 0.0003,
    },
]


def get_ccla_config(name, hf_cache_dir: str, data_dir: str):
    base = {
        "seq_length": 2048,
        "max_position_embeddings": 2048,
        "global_batch_size": 1024,
        "norm_epsilon": 1e-5,
        "normalization": "RMSNorm",
        "position_embedding_type": "rope",
        "rotary_base": 10000,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "weight_decay": 0.01,
        "transformer_impl": "transformer_engine",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-8,
        "tokenizer_type": "HFTokenizer",
        "vocab_file": hf_hub_download("EleutherAI/pythia-70m", "tokenizer.json", revision="main"),
        "data_path": Path(data_dir) / "slimpj" / "slimpj-neox-c1c2_text_document",
        "init_method_std": 0.02,
    }
    for config in ccla_configs:
        if config['name'] == name:
            return config | base
    return None
