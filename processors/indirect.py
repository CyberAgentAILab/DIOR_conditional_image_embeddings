import glob
import json
import os
import time
from typing import Any, List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from entities.config import InferenceConfig
from entities.indirect_texts import InDiReCTTexts
from utils.get_models import CLIPTextEncoder


class InDiReCTEmbeddingProcessor:
    """InDiReCT baseline: CLIP + DimRedRecon transformation.

    Uses text descriptions to learn a transformation matrix that projects
    CLIP image embeddings into a conditional space.
    """

    def __init__(
        self,
        config: InferenceConfig,
        clip_model_id: str = "openai/clip-vit-base-patch32",
        num_components: int = 128,
        patience: int = 100,
        learning_rate: float = 0.01,
    ):
        self.config = config
        self.clip_model_id = clip_model_id
        self.num_components = num_components
        self.patience = patience
        self.learning_rate = learning_rate

        self.dataset_dir = f"{config.datasets_dir}/{config.dataset_name}"

        self.clip_encoder = CLIPTextEncoder(
            model_id=clip_model_id,
            torch_dtype=torch.float32,
        )

        self.dimred = None
        self._text_features = None

    def run(self) -> None:
        """Run InDiReCT embedding extraction for all aspects."""
        dimred_cls = self._load_dim_red_recon()
        image_files = self._get_image_files()

        aspects = self._get_available_aspects()
        if not aspects:
            print(f"No InDiReCT text configs for dataset: {self.config.dataset_name}")
            return

        start_time = time.time()

        print("Loading images and computing CLIP features...")
        images, valid_paths = self._load_all_images(image_files)
        if not images:
            print("No images found")
            return

        image_features = self.clip_encoder.encode_images(images, batch_size=32, normalize=True, show_progress=True)

        for aspect in aspects:
            print(f"\nProcessing aspect: {aspect}")
            self._process_aspect(aspect, image_features, valid_paths, dimred_cls)

        elapsed = time.time() - start_time
        print(f"\nProcessed {aspects} in {elapsed:.2f}s")

    def _load_dim_red_recon(self) -> type[Any]:
        """Load DimRedRecon from the optional external InDiReCT checkout."""
        try:
            from utils.indirect.dimensionality_reduction import DimRedRecon
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("utils.indirect"):
                raise SystemExit(
                    "InDiReCT is not cloned. Run `uv run poe indirect-setup` "
                    "or clone https://github.com/LSX-UniWue/InDiReCT.git into utils/indirect."
                ) from None
            raise
        return DimRedRecon

    def _get_available_aspects(self) -> List[str]:
        """Get aspects that have InDiReCT text configs."""
        available = []
        for dataset, aspect in InDiReCTTexts.list_configs():
            if dataset == self.config.dataset_name:
                available.append(aspect)
        return available

    def _process_aspect(
        self,
        aspect: str,
        image_features: torch.Tensor,
        image_paths: List[str],
        dimred_cls: type[Any],
    ) -> None:
        """Process a single aspect: fit DimRedRecon and save embeddings."""
        text_config = InDiReCTTexts.get(self.config.dataset_name, aspect)
        if text_config is None:
            print(f"Skipping {aspect}: no InDiReCT text config")
            return

        texts = text_config.get_texts()
        print(f"  Computing text features for {len(texts)} descriptions...")
        text_features = self.clip_encoder.encode_text(texts, batch_size=32, normalize=True)

        print(f"  Fitting DimRedRecon (num_components={self.num_components})...")
        dimred = dimred_cls(
            num_components=self.num_components,
            patience=self.patience,
            learning_rate=self.learning_rate,
        )
        dimred.fit(text_features)

        print("  Transforming image features...")
        transformed = dimred.transform(image_features)
        transformed = transformed.detach().cpu().to(torch.float32)

        print("  Saving embeddings...")
        self._save_embeddings(transformed, image_paths, aspect)

    def _save_embeddings(
        self,
        embeddings: torch.Tensor,
        image_paths: List[str],
        aspect: str,
    ) -> None:
        """Save transformed embeddings to JSON files."""
        out_dir = self._build_output_dir(aspect)
        os.makedirs(out_dir, exist_ok=True)

        for i, path in enumerate(image_paths):
            filename = os.path.splitext(os.path.basename(path))[0] + ".json"
            output_path = os.path.join(out_dir, filename)

            data = {
                "embedding": embeddings[i].numpy().tolist(),
                "source": "indirect",
                "clip_model": self.clip_model_id,
                "num_components": self.num_components,
                "aspect": aspect,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")

    def _build_output_dir(self, aspect: str) -> str:
        """Build output directory path."""
        model_name = self.clip_model_id.replace("/", "_")
        aspect_str = aspect.replace(" ", "_")
        return (
            f"{self.config.embedding_dir}/"
            f"{self.config.dataset_name}-{model_name}-indirect-{aspect_str}"
            f"-num_components={self.num_components}"
        )

    def _get_image_files(self) -> List[str]:
        """Get list of image files to process."""
        if self.config.dataset_info_dir:
            info_path = f"{self.config.dataset_info_dir}/info_files/{self.config.dataset_name}/test.json"
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            files = [os.path.join(self.config.dataset_info_dir, x["path"]) for x in data]
        else:
            files = sorted(glob.glob(f"{self.dataset_dir}/*"))

        return [p for p in files if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

    def _load_all_images(self, image_files: List[str]) -> tuple[List[Image.Image], List[str]]:
        """Load all images and return valid ones."""
        images, valid_paths = [], []

        for path in tqdm(image_files, desc="Loading images"):
            img = self._load_image(path)
            if img is not None:
                images.append(img)
                valid_paths.append(path)

        return images, valid_paths

    def _load_image(self, path: str) -> Optional[Image.Image]:
        """Load and resize image."""
        if not os.path.exists(path):
            return None

        try:
            image = Image.open(path).convert("RGB")
            w, h = image.size
            max_size = self.config.image_max_size

            if max(w, h) > max_size:
                if w > h:
                    image = image.resize((max_size, int(max_size * h / w)), Image.Resampling.LANCZOS)
                else:
                    image = image.resize((int(max_size * w / h), max_size), Image.Resampling.LANCZOS)

            return image
        except Exception:
            return None
