#!/bin/bash
# CLIP Embedding (Baseline)

uv run python inference.py \
    --model_id openai/clip-vit-large-patch14 \
    --dataset_name cub200 \
    --prompt_type describe
