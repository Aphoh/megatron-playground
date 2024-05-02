from dataclasses import dataclass
from transformers import OlmoConfig, LlamaConfig, GPTNeoXConfig, GPT2Config, GPT2Tokenizer
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Optional
from tokenizers import Tokenizer
import torch

@dataclass
class ModelDescriptor:
    num_layers: int
    hidden_size: int
    seq_length: int
    num_attention_heads: int
    max_position_embeddings: int
    position_embedding_type: str
    tokenizer_type: str
    make_vocab_size_divisible_by: Optional[int]
    params_dtype: torch.dtype
    output_layer: bool
    linear_bias: bool
    qkv_bias: bool
    model_type: str
    bert_binary_head: Optional[bool]
    true_vocab_size: Optional[int]
    norm_has_weight: bool
    norm_has_bias: bool
    glu: bool
    checkpoint_args: dict
    consumed_train_samples: Optional[int]
    consumed_valid_samples: Optional[int]
    iteration: Optional[int]

    previous_tensor_parallel_size: Optional[int] = 1
    previous_pipeline_parallel_size: Optional[int] = 1

def get_llama_args(repo: str, revision: str, cache_dir: str, data_dir: str, load_type: str, set_args=None) -> dict:
    res = {}
    config = LlamaConfig.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    tokenizer_file = hf_hub_download(repo, "tokenizer.json", revision=revision, cache_dir=cache_dir)
    res["seq_length"] = config.max_position_embeddings
    res["max_position_embeddings"] = config.max_position_embeddings
    res["hidden_size"] = config.hidden_size
    res["num_attention_heads"] = config.num_attention_heads
    res["num_layers"] = config.num_hidden_layers
    res["global_batch_size"] = 1024
    res["norm_epsilon"] = "%f" % config.rms_norm_eps if set_args else config.rms_norm_eps
    res["position_embedding_type"] = "rope"
    if set_args:
        res["iteration"] = 1
    res["rotary_base"] = int(config.rope_theta)
    res["act_fn"] = "silu"
    res["glu"] = () if set_args else True
    res["tokenizer_type"] = "HFTokenizer"
    res["vocab_file"] = tokenizer_file
    res["bf16"] = () if set_args else True
    res["normalization"] = "RMSNorm"
    if set_args:
        res["disable_bias_linear"] = ()
        res["untie_embeddings_and_output_weights"] = ()
    else:
        res["add_bias_linear"] = False
        res["untie_embeddings_and_output_weights"] = True

    res["ffn_hidden_size"] = config.intermediate_size
    if hasattr(config, "num_key_value_heads"):
        res["group_query_attention"] = () if set_args else True
        res["num_query_groups"] = config.num_key_value_heads

    res["vocab_size"] = Tokenizer.from_file(tokenizer_file).get_vocab_size()
    file = f"slimpj-{load_type}-c1_text_document"
    res["data_path"] = Path(data_dir) / "slimpj" / file
    res["attention_dropout"] = 0.0
    res["hidden_dropout"] = 0.0
    res["weight_decay"] = 0.01
    res["transformer_impl"] = "transformer_engine"

    # training args
    res["adam_beta1"] = 0.9
    res["adam_beta2"] = 0.95
    res["adam_eps"] = 1e-5
    res["global_batch_size"] = 1024
    if set_args:
        for k, v in res.items():
            setattr(set_args, k, v)
    
    return res

def get_pythia_args(repo: str, revision: str, cache_dir: str, data_dir: str, load_type: str, set_args=None) -> dict:
    res = {}
    pythia_config = GPTNeoXConfig.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    tokenizer_path = hf_hub_download(repo, "tokenizer.json", revision=revision, cache_dir=cache_dir)
    # Arguments from the config
    res["seq_length"] = pythia_config.max_position_embeddings
    res["max_position_embeddings"] = pythia_config.max_position_embeddings
    res["hidden_size"] = pythia_config.hidden_size
    res["num_attention_heads"] = pythia_config.num_attention_heads
    res["num_layers"] = pythia_config.num_hidden_layers
    res["global_batch_size"] = 1024
    res["norm_epsilon"] = pythia_config.layer_norm_eps
    if set_args:
        res["iteration"] = 1
    res["position_embedding_type"] = "rope"
    res["rotary_percent"] = pythia_config.rotary_pct
    if pythia_config.use_parallel_residual:
        res["use_parallel_residual"] = () if set_args else True
    res["tokenizer_type"] = "HFTokenizer"
    res["normalization"] = "LayerNorm"
    if not pythia_config.tie_word_embeddings:
        res["untie_embeddings_and_output_weights"] = () if set_args else True
    res["vocab_size"] = Tokenizer.from_file(tokenizer_path).get_vocab_size()
    res["vocab_file"] = tokenizer_path
    res["ffn_hidden_size"] = pythia_config.intermediate_size
    res["init_method_std"] = pythia_config.initializer_range
    res["max_position_embeddings"] = pythia_config.max_position_embeddings
    assert pythia_config.hidden_act == "gelu"
    res["act_fn"] = "gelu"
    res["hidden_dropout"] = 0.0
    res["attention_dropout"] = 0.0
    res["weight_decay"] = 0.01

    # download tokenizer
    res["data_path"] = Path(data_dir) / "slimpj" / "slimpj-neox-c1c2_text_document"
    res["seq_length"] = 2048
    res["transformer_impl"] = "transformer_engine"

    # Training args
    res["adam_beta1"] = 0.9
    res["adam_beta2"] = 0.95
    res["adam_eps"] = 1e-8
    res["global_batch_size"] = 1024

    assert pythia_config.attention_bias
    return res

def get_olmo_args(repo: str, revision: str, cache_dir: str, load_type: str, set_args=None, data_dir=None):
    config = OlmoConfig.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    tokenizer_file = hf_hub_download(
        repo, "tokenizer.json", revision=revision, cache_dir=cache_dir
    )
    tokenizer = Tokenizer.from_file(tokenizer_file)
    res = {}
    true_val = () if set_args else True
    # Update Megatron args.
    res["seq_length"] = config.max_sequence_length
    res["max_position_embeddings"] = config.max_sequence_length
    res["hidden_size"] = config.d_model
    res["num_attention_heads"] = config.n_heads
    res["num_layers"] = config.n_layers
    res["global_batch_size"] = 2048
    res["norm_epsilon"] = 1e-5
    res["position_embedding_type"] = "rope"
    res["rotary_base"] = config.rope_theta
    res["act_fn"] = 'silu'
    if config.mlp_hidden_size:
        res["ffn_hidden_size"] = config.mlp_hidden_size // 2
    else:
        res["ffn_hidden_size"] = config.d_model * config.mlp_ratio // 2
    assert config.activation_type == 'swiglu'
    res["glu"] = true_val
    res["tokenizer_type"] = "HFTokenizer"
    res["vocab_size"] = config.vocab_size
    assert tokenizer.get_vocab_size() == config.vocab_size
    res["vocab_file"] = tokenizer_file
    res["bf16"] = true_val
    res["normalization"] = "NonParametricLayerNorm"
    if set_args:
        res["disable_bias_linear"] = ()
    else:
        res["iteration"] = 1  # '0', 'release' don't work
        res["add_bias_linear"] = False
    if not config.weight_tying:
        res["untie_embeddings_and_output_weights"] = () if set_args else True
    res["vocab_size"] = tokenizer.get_vocab_size()
    if data_dir:
        res["data_path"] = Path(data_dir) / "slimpj" / "slimpj-neox-c1c2_text_document"
    res["attention_dropout"] = 0.0
    res["hidden_dropout"] = 0.0
    res["weight_decay"] = 0.1 # TODO: is this optimal?
    res["transformer_impl"] = "transformer_engine"

    # Training args
    res["adam_beta1"] = 0.9
    res["adam_beta2"] = 0.95
    res["adam_eps"] = 1e-5
    res["global_batch_size"] = 2048
    if set_args:
        for k, v in res.items():
            setattr(set_args, k, v)
    return res

def get_gpt2_args(repo: str, revision: str, cache_dir: str, data_dir: str, load_type: str, set_args=None) -> dict:
    res = {}
    config = GPT2Config.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    merge_file = hf_hub_download(repo, "merges.txt", revision=revision, cache_dir=cache_dir)
    vocab_file = hf_hub_download(repo, "vocab.json", revision=revision, cache_dir=cache_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    # Arguments from the config
    res["seq_length"] = config.n_positions
    res["max_position_embeddings"] = config.n_positions
    res["hidden_size"] = config.n_embd
    res["num_attention_heads"] = config.n_head
    res["num_layers"] = config.n_layer
    res["global_batch_size"] = 1024
    res["norm_epsilon"] = config.layer_norm_epsilon
    if set_args:
        res["iteration"] = 1
    res["position_embedding_type"] = "learned_absolute"
    if config.use_parallel_residual:
        res["use_parallel_residual"] = () if set_args else True
    res["tokenizer_type"] = "HFTokenizer"
    res["normalization"] = "LayerNorm"
    if not config.tie_word_embeddings:
        res["untie_embeddings_and_output_weights"] = () if set_args else True
    res["vocab_size"] = config.vocab_size
    assert tokenizer.vocab_size == config.vocab_size
    res["vocab_file"] = vocab_file
    res["merge_file"] = merge_file
    res["ffn_hidden_size"] = config.intermediate_size
    res["init_method_std"] = config.initializer_range
    res["max_position_embeddings"] = config.max_position_embeddings
    assert config.hidden_act == "gelu"
    res["act_fn"] = "gelu"
    res["hidden_dropout"] = 0.0
    res["attention_dropout"] = 0.0
    res["weight_decay"] = 0.01

    # download tokenizer
    res["data_path"] = Path(data_dir) / "slimpj" / "slimpj-neox-c1c2_text_document"
    res["seq_length"] = 2048
    res["transformer_impl"] = "transformer_engine"

    # Training args
    res["adam_beta1"] = 0.9
    res["adam_beta2"] = 0.95
    res["adam_eps"] = 1e-8
    res["global_batch_size"] = 1024

    assert config.attention_bias
    return res


ARG_SETTERS = {
    "pythia": get_pythia_args,
    "llama": get_llama_args,
    "llama3": get_llama_args,
    "olmo": get_olmo_args,
}