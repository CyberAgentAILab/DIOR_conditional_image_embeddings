import argparse
import random
from typing import Optional, Protocol

import numpy as np
import torch

from entities.config import InferenceConfig
from processors import (
    CLIPEmbeddingProcessor,
    DIOREmbeddingProcessor,
    GenerativeEmbeddingProcessor,
    InDiReCTEmbeddingProcessor,
)
from utils.get_models import is_clip_model

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EmbeddingProcessor(Protocol):
    def run(self) -> None:
        ...


def create_processor(
    config: InferenceConfig,
    generative: bool = False,
    text_encoder_id: Optional[str] = None,
    indirect: bool = False,
    indirect_num_components: int = 128,
) -> EmbeddingProcessor:
    if indirect:
        return InDiReCTEmbeddingProcessor(
            config,
            clip_model_id=config.model_id,
            num_components=indirect_num_components,
        )
    if generative:
        return GenerativeEmbeddingProcessor(config, text_encoder_id)
    if is_clip_model(config.model_id):
        return CLIPEmbeddingProcessor(config)
    return DIOREmbeddingProcessor(config)


def parse_args() -> tuple:
    parser = argparse.ArgumentParser(description="Embedding extraction")
    parser.add_argument("--model_id", "--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--dataset_info_dir", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default="describe")
    parser.add_argument("--embedding_dir", type=str, default="./embeddings")
    parser.add_argument("--num_layer", type=int, default=-1)
    parser.add_argument("--num_token", type=int, default=-1)
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--image_max_size", type=int, default=700)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--embedding_source", type=str, default="input_last", choices=["input_last", "first_output", "mean_output"]
    )
    parser.add_argument("--gen_max_new_tokens", type=int, default=16)
    parser.add_argument("--gen_do_sample", action="store_true")
    parser.add_argument("--gen_temperature", type=float, default=0.0)
    parser.add_argument(
        "--generative",
        action="store_true",
        help="Use generative mode: VLM generates text, then encode with text encoder",
    )
    parser.add_argument(
        "--text_encoder_id",
        type=str,
        default=None,
        help="Text encoder model ID for generative mode (default: sentence-t5-base)",
    )
    parser.add_argument(
        "--indirect", action="store_true", help="Use InDiReCT baseline: CLIP + DimRedRecon transformation"
    )
    parser.add_argument(
        "--indirect_num_components", type=int, default=128, help="Number of components for DimRedRecon (default: 128)"
    )

    args = parser.parse_args()

    config = InferenceConfig(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        datasets_dir=args.datasets_dir,
        dataset_info_dir=args.dataset_info_dir,
        prompt_type=args.prompt_type,
        embedding_dir=args.embedding_dir,
        num_layer=args.num_layer,
        num_token=args.num_token,
        save_all=args.save_all,
        image_max_size=args.image_max_size,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        embedding_source=args.embedding_source,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_do_sample=args.gen_do_sample,
        gen_temperature=args.gen_temperature,
    )

    return config, args


if __name__ == "__main__":
    set_seed()
    config, args = parse_args()
    processor = create_processor(
        config,
        generative=args.generative,
        text_encoder_id=args.text_encoder_id,
        indirect=args.indirect,
        indirect_num_components=args.indirect_num_components,
    )
    processor.run()
