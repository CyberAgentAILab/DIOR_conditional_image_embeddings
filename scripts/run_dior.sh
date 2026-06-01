#!/bin/bash
# DIOR (Proposed Method)

uv run python inference.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name cub200 \
    --prompt_type describe \
    --num_layer -1 \
    --num_token -1
