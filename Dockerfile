FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install wandb transformers tokenizers torchist