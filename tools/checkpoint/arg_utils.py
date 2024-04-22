from dataclasses import dataclass
from typing import Optional
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
    bias_linear: bool
    qkv_bias: bool
    model_type: str
    bert_binary_head: Optional[bool]
    true_vocab_size: Optional[int]
    norm_has_bias: bool
    glu: bool
    checkpoint_args: dict
    consumed_train_samples: Optional[int]
    consumed_valid_samples: Optional[int]
    iteration: Optional[int]

    previous_tensor_parallel_size: Optional[int] = 1
    previous_pipeline_parallel_size: Optional[int] = 1