#!/bin/bash
# Generative Mode (VLM -> Text -> Sentence-T5)

uv run python inference.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name cub200 \
    --prompt_type describe \
    --generative \
    --text_encoder_id sentence-transformers/sentence-t5-base
