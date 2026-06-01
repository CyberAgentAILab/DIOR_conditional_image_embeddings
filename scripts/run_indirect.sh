#!/bin/bash
# InDiReCT Baseline

uv run python inference.py \
    --model_id openai/clip-vit-large-patch14 \
    --dataset_name cub200 \
    --indirect \
    --indirect_num_components 128
