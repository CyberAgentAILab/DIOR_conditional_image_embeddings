#!/bin/bash
# Evaluation

uv run python evaluate.py \
    --embedding_dir ./embeddings \
    --setting_pattern "cub200-*"
