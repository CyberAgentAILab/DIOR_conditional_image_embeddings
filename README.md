# Training-free Conditional Image Embedding Framework Leveraging Large Vision Language Models

[![WACV2026](https://img.shields.io/badge/WACV-2026-blue)](https://wacv.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-WACV2026-red)](https://openaccess.thecvf.com/content/WACV2026/html/Kawarada_Training-free_Conditional_Image_Embedding_Framework_Leveraging_Large_Vision_Language_Models_WACV_2026_paper.html)

**Accepted to WACV 2026**

> Conditional image embeddings are feature representations that focus on specific aspects of an image indicated by a given textual condition (e.g., color, genre). DIOR is a training-free approach that prompts a Large Vision-Language Model (LVLM) to describe an image with a single word related to a given condition. The hidden state vector of the LVLM's last token is then extracted as the conditional image embedding.

## Overview

DIOR provides a versatile solution that can be applied to any image and condition without additional training or task-specific priors. Comprehensive experimental results on conditional image similarity tasks demonstrate that DIOR outperforms existing training-free baselines, including [CLIP](https://arxiv.org/abs/2103.00020), and achieves superior performance compared to methods that require additional training across multiple settings.

## Embedding Processors

| Processor | Description |
|-----------|-------------|
| `DIOREmbeddingProcessor` | **DIOR** - Extract embeddings from VLM hidden states with KV cache support |
| `CLIPEmbeddingProcessor` | Extract embeddings from [CLIP](https://arxiv.org/abs/2103.00020) models |
| `GenerativeEmbeddingProcessor` | VLM generates text -> Text encoder ([Sentence-T5](https://arxiv.org/abs/2108.08877)) creates embeddings |
| `InDiReCTEmbeddingProcessor` | [CLIP](https://arxiv.org/abs/2103.00020) + DimRedRecon transformation (baseline method) |

## Installation

```bash
uv sync
```

For development tools:

```bash
uv sync --group dev
```

## Dataset Preparation

Download the datasets used in our experiments and place them under [`datasets/`](datasets/). Each dataset directory must contain image files **directly** (no class subdirectories) — `evaluate.py` recovers labels from filenames / CSV metadata.

### One-shot download script

```bash
bash scripts/download_datasets.sh
# or target a custom location:
bash scripts/download_datasets.sh /path/to/datasets
```

This automatically downloads and flattens `cub200` and `movie_posters`. For datasets that require authentication or manual terms-of-use acceptance (`cars196`, `deepfashion`), the script prints the steps you need to follow.

### Dataset sources

- **Synthetic Cars** (`synthetic_cars`) — **the original download link ([`konstantinkobs/DML-analysis`](https://github.com/konstantinkobs/DML-analysis/tree/master/3D_cars)) is no longer accessible.** You will need to obtain the images from another mirror and place them (flat) under `datasets/synthetic_cars/`.
- **Stanford Cars** (`cars196`) — [Cars196 on Kaggle](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset) (requires a Kaggle account).
- **DeepFashion In-Shop** (`deepfashion`) — [InShop](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) (requires terms acceptance). Download `img.zip` from the [In-Shop folder on Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pekpRNUlMS3Z5cUk), place it at `datasets/deepfashion/img.zip`, and re-run `scripts/download_datasets.sh` — the script will extract and flatten the images to match `metadata/deepfashon_metadata.csv` (4,000 test images).
- **CUB-200-2011** (`cub200`) — [CUB200](https://data.caltech.edu/records/65de6-vp158) (public, fully automated).
- **Movie Posters** (`movie_posters`) — [Movie Posters](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/) (public, fully automated).

After setup, the directory structure should look like:

```
datasets/
├── synthetic_cars/
├── cub200/
├── movie_posters/
├── deepfashion/
└── cars196/
```

## Usage

### DIOR (Proposed Method)

```bash
uv run python inference.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name cub200 \
    --prompt_type describe \
    --num_layer -1 \
    --num_token -1
```

### With KV Cache (faster for multiple aspects)

```bash
uv run python inference.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name cub200 \
    --prompt_type describe \
    --use_cache
```

### CLIP Embedding (Baseline)

```bash
uv run python inference.py \
    --model_id openai/clip-vit-large-patch14 \
    --dataset_name cub200 \
    --prompt_type describe
```

### Generative Mode (VLM -> Text -> Sentence-T5)

```bash
uv run python inference.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name cub200 \
    --prompt_type describe \
    --generative \
    --text_encoder_id sentence-transformers/sentence-t5-base
```

### InDiReCT Baseline

InDiReCT is an optional external dependency. Clone the original implementation
before running `--indirect`:

```bash
uv run poe indirect-setup
```

If the upstream repository changes, override the clone source:

```bash
INDIRECT_REPO_URL=https://github.com/<owner>/<repo>.git uv run poe indirect-setup
```

```bash
uv run python inference.py \
    --model_id openai/clip-vit-large-patch14 \
    --dataset_name cub200 \
    --indirect \
    --indirect_num_components 128
```

## Evaluation

```bash
uv run python evaluate.py \
    --embedding_dir ./embeddings \
    --setting_pattern "cub200-*"
```

## Development

```bash
uv run ruff check .
uv run mypy
```

## Supported Datasets

- `synthetic_cars` - Synthetic car images (car_model, car_color, background_color)
- `cars196` - Stanford Cars dataset (car_model only; the `manufacturer` and `car_type` aspects reported in the paper are not evaluable here because the corresponding metadata is not publicly available from prior work)
- `cub200` - CUB-200-2011 bird dataset (bird_species)
- `movie_posters` - Movie poster dataset (genre, country)
- `deepfashion` - DeepFashion dataset (clothing_category, texture, fabric, fit)

## Project Structure

```
conditional-image-embeddings/
├── inference.py              # Main entry point
├── evaluate.py               # Evaluation script
├── processors/
│   ├── base.py               # BaseEmbeddingProcessor
│   ├── clip.py               # CLIPEmbeddingProcessor
│   ├── dior.py               # DIOREmbeddingProcessor (Proposed)
│   ├── generative.py         # GenerativeEmbeddingProcessor
│   └── indirect.py           # InDiReCTEmbeddingProcessor
├── entities/
│   ├── config.py             # InferenceConfig, TextEncoder
│   ├── dataset.py            # Dataset configurations
│   ├── prompt.py             # Prompt templates
│   ├── embedding.py          # EmbeddingOutput
│   └── indirect_texts.py     # InDiReCT text descriptions
├── utils/
│   ├── get_models.py         # Model loading utilities
│   └── utils.py              # Helper functions
├── utils/indirect/           # Optional external InDiReCT clone (gitignored)
│   └── dimensionality_reduction.py  # DimRedRecon from InDiReCT
├── metadata/                 # Dataset metadata files
├── datasets/                 # Image datasets
└── embeddings/               # Output embeddings
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@InProceedings{Kawarada_2026_WACV,
    author    = {Kawarada, Masayuki and Yamada, Kosuke and Tejero-de-Pablos, Antonio and Inoue, Naoto},
    title     = {Training-free Conditional Image Embedding Framework Leveraging Large Vision Language Models},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {7636-7646}
}
```

## References

- InDiReCT: [Indirect Dimensionality Reduction for Conditional Embeddings](https://github.com/LSX-UniWue/InDiReCT) ([paper](https://arxiv.org/abs/2211.12760))
- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Sentence-T5: [Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models](https://arxiv.org/abs/2108.08877)
- Qwen2.5-VL: [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)

## License

This project is released under the [MIT License](LICENSE).
