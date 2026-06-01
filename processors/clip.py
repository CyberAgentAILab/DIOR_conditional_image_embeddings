import os
from typing import List, Optional

import torch
from PIL import Image

from .base import BaseEmbeddingProcessor


class CLIPEmbeddingProcessor(BaseEmbeddingProcessor):
    """Embedding extraction for CLIP models."""

    def _process_single_image(self, image_path: str) -> None:
        image = self._load_image(image_path)
        if image is None:
            return

        filename = self._get_json_filename(image_path)
        output_path = self._build_output_path(filename, None)

        if self._output_exists(output_path):
            return

        self._extract_and_save(image, output_path)
        torch.cuda.empty_cache()

    def _process_batch(self, image_paths: List[str], aspect: Optional[str]) -> None:
        images, valid_paths = self._load_images(image_paths)
        if not images:
            return

        filenames = [self._get_json_filename(p) for p in valid_paths]
        output_paths = [self._build_output_path(f, None) for f in filenames]

        to_process = [(img, path) for img, path in zip(images, output_paths) if not self._output_exists(path)]

        if not to_process:
            return

        images_to_process = [x[0] for x in to_process]
        paths_to_process = [x[1] for x in to_process]

        self._extract_and_save_batch(images_to_process, paths_to_process)
        torch.cuda.empty_cache()

    def _extract_and_save(self, image: Image.Image, output_path: str) -> None:
        inputs = self._to_device(self.processor(images=image, return_tensors="pt"))

        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                embeds = self.model.get_image_features(**inputs)
            else:
                embeds = self.model.encode_image(inputs["pixel_values"])

        self._save_embedding(embeds, output_path)

    def _extract_and_save_batch(self, images: List[Image.Image], output_paths: List[str]) -> None:
        inputs = self._to_device(self.processor(images=images, return_tensors="pt"))

        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                embeds = self.model.get_image_features(**inputs)
            else:
                embeds = self.model.encode_image(inputs["pixel_values"])

        self._save_embedding_batch(embeds, output_paths)

    def _save_embedding(self, embeds: torch.Tensor, output_path: str) -> None:
        data = embeds.detach().cpu().to(torch.float32).numpy().tolist()
        self._save_json(output_path, {"embedding": data, "source": "clip"})

    def _save_embedding_batch(self, embeds: torch.Tensor, output_paths: List[str]) -> None:
        embeds = embeds.detach().cpu().to(torch.float32)
        for i, path in enumerate(output_paths):
            self._save_json(path, {"embedding": embeds[i].numpy().tolist(), "source": "clip"})

    def _build_output_path(self, filename: str, aspect: Optional[str]) -> str:
        out_dir = self._get_base_output_dir() + "-embedsrc=clip"
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, filename)

    def _output_exists(self, output_path: str) -> bool:
        return os.path.exists(output_path)
