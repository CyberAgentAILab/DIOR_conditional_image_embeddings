import glob
import json
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from entities.config import InferenceConfig
from entities.dataset import Datasets
from utils.get_models import load_model_and_processor


def chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class BaseEmbeddingProcessor(ABC):
    """Base class for embedding extraction from images."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model, self.processor = load_model_and_processor(
            model_id=config.model_id,
        )
        self.aspects = Datasets.get(config.dataset_name).aspects
        self.dataset_dir = f"{config.datasets_dir}/{config.dataset_name}"
        self.model.eval()

    def run(self) -> None:
        image_files = self._get_image_files()

        start_time = time.time()

        if self.config.batch_size == 1:
            self._run_single_mode(image_files)
        else:
            self._run_batch_mode(image_files)

        elapsed = time.time() - start_time
        print(f"Processed {self.aspects} (batch={self.config.batch_size}) in {elapsed:.2f}s")

    def _run_single_mode(self, image_files: List[str]) -> None:
        for path in tqdm(image_files):
            self._process_single_image(path)

    def _run_batch_mode(self, image_files: List[str]) -> None:
        for aspect in self.aspects:
            batches = list(chunked(image_files, self.config.batch_size))
            for batch in tqdm(batches, desc=f"Aspect: {aspect}"):
                self._process_batch(batch, aspect)

    @abstractmethod
    def _process_single_image(self, image_path: str) -> None:
        pass

    @abstractmethod
    def _process_batch(self, image_paths: List[str], aspect: Optional[str]) -> None:
        pass

    @abstractmethod
    def _build_output_path(self, filename: str, aspect: Optional[str]) -> str:
        pass

    @abstractmethod
    def _output_exists(self, output_path: str) -> bool:
        pass

    def _load_image(self, path: str) -> Optional[Image.Image]:
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            return None

        image = Image.open(path).convert("RGB")
        w, h = image.size
        max_size = self.config.image_max_size

        if max(w, h) > max_size:
            if w > h:
                image = image.resize((max_size, int(max_size * h / w)), Image.Resampling.LANCZOS)
            else:
                image = image.resize((int(max_size * w / h), max_size), Image.Resampling.LANCZOS)

        return image

    def _load_images(self, paths: List[str]) -> tuple:
        images, valid_paths = [], []
        for p in paths:
            img = self._load_image(p)
            if img:
                images.append(img)
                valid_paths.append(p)
        return images, valid_paths

    def _get_image_files(self) -> List[str]:
        if self.config.dataset_info_dir:
            info_path = f"{self.config.dataset_info_dir}/info_files/{self.config.dataset_name}/test.json"
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            files = [os.path.join(self.config.dataset_info_dir, x["path"]) for x in data]
        else:
            files = sorted(glob.glob(f"{self.dataset_dir}/*"))

        return [p for p in files if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

    def _get_json_filename(self, image_path: str) -> str:
        return os.path.splitext(os.path.basename(image_path))[0] + ".json"

    def _get_base_output_dir(self) -> str:
        model_name = self.config.model_id.replace("/", "_")
        return f"{self.config.embedding_dir}/{self.config.dataset_name}-{model_name}"

    def _to_device(self, inputs: dict) -> dict:
        device, dtype = self._get_device_and_dtype()
        return {
            k: v.to(device).to(dtype)
            if torch.is_tensor(v) and v.dtype.is_floating_point
            else v.to(device)
            if torch.is_tensor(v)
            else v
            for k, v in inputs.items()
        }

    def _get_device_and_dtype(self) -> tuple:
        param = next(self.model.parameters())
        return param.device, param.dtype

    def _save_json(self, path: str, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
