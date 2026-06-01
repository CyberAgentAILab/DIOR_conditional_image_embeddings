#!/bin/bash
# DIOR with KV Cache (faster for multiple aspects)

uv run python inference.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name cub200 \
    --prompt_type describe \
    --use_cache
