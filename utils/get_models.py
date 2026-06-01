from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    ProcessorMixin,
)


def is_clip_model(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    return "clip" in model_id_lower and "llava" not in model_id_lower


class BaseModelLoader(ABC):
    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        self._model: Optional[PreTrainedModel] = None
        self._processor: Optional[ProcessorMixin] = None

    def _get_common_kwargs(self) -> dict:
        return {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "device_map": "auto",
        }

    def _load_processor(self) -> None:
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

    @abstractmethod
    def _load_model(self) -> None:
        pass

    def load(self) -> Tuple[PreTrainedModel, ProcessorMixin]:
        self._load_processor()
        self._load_model()
        return self._model, self._processor


class CLIPModelLoader(BaseModelLoader):
    def _load_model(self) -> None:
        self._model = AutoModel.from_pretrained(self.model_id, **self._get_common_kwargs())


class VLMModelLoader(BaseModelLoader):
    def _load_model(self) -> None:
        self._model = AutoModelForVision2Seq.from_pretrained(self.model_id, **self._get_common_kwargs())
        self._force_left_padding()

    def _force_left_padding(self) -> None:
        tok = getattr(self._processor, "tokenizer", None)
        if tok is None:
            return

        tok.padding_side = "left"

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token

        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.pad_token_id = tok.pad_token_id

        if getattr(getattr(self._model, "config", None), "pad_token_id", None) is None:
            self._model.config.pad_token_id = tok.pad_token_id


def load_model_and_processor(
    model_id: str,
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PreTrainedModel, ProcessorMixin]:
    loader_cls = CLIPModelLoader if is_clip_model(model_id) else VLMModelLoader
    loader = loader_cls(
        model_id=model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    return loader.load()


class CLIPTextEncoder:
    """Encode text using CLIP-like models via transformers."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = True,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._image_processor = None

    def _load(self) -> None:
        if self._model is not None:
            return

        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)
        self._model.eval()

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            self._tokenizer = getattr(self._processor, "tokenizer", None)
            if self._tokenizer is None:
                self._tokenizer = self._processor
            self._image_processor = getattr(self._processor, "image_processor", None)
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            self._image_processor = None

        if self._image_processor is None:
            from transformers import CLIPImageProcessor

            if "EVA" in self.model_id.upper():
                if "18B" in self.model_id or "8B" in self.model_id:
                    self._image_processor = CLIPImageProcessor(
                        size={"shortest_edge": 224},
                        crop_size={"height": 224, "width": 224},
                        image_mean=[0.48145466, 0.4578275, 0.40821073],
                        image_std=[0.26862954, 0.26130258, 0.27577711],
                    )
                else:
                    self._image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            else:
                try:
                    self._image_processor = CLIPImageProcessor.from_pretrained(self.model_id)
                except Exception:
                    try:
                        from transformers import AutoImageProcessor

                        self._image_processor = AutoImageProcessor.from_pretrained(
                            self.model_id, trust_remote_code=self.trust_remote_code
                        )
                    except Exception:
                        self._image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @property
    def model(self) -> PreTrainedModel:
        self._load()
        return self._model

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    def encode_text(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Encode texts into CLIP text embeddings."""
        self._load()

        all_features = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + batch_size]

                inputs = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                if hasattr(self._model, "get_text_features"):
                    features = self._model.get_text_features(**inputs)
                elif hasattr(self._model, "encode_text"):
                    features = self._model.encode_text(**inputs)
                else:
                    outputs = self._model.text_model(**inputs)
                    features = outputs.pooler_output
                    if features is None:
                        features = outputs.last_hidden_state[:, 0, :]

                all_features.append(features.cpu())

        features = torch.cat(all_features, dim=0)

        if normalize:
            features = features / features.norm(dim=1, keepdim=True)

        return features.to(torch.float32)

    def encode_images(
        self,
        images: List,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Encode images into CLIP image embeddings."""
        self._load()

        if self._image_processor is None:
            raise ValueError(f"No image processor available for {self.model_id}")

        all_features = []

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images")

        with torch.no_grad():
            for i in iterator:
                batch_images = images[i : i + batch_size]

                inputs = self._image_processor(
                    images=batch_images,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}

                if hasattr(self._model, "get_image_features"):
                    features = self._model.get_image_features(**inputs)
                elif hasattr(self._model, "encode_image"):
                    features = self._model.encode_image(inputs["pixel_values"])
                else:
                    outputs = self._model.vision_model(**inputs)
                    features = outputs.pooler_output
                    if features is None:
                        features = outputs.last_hidden_state[:, 0, :]

                all_features.append(features.cpu())

        features = torch.cat(all_features, dim=0)

        if normalize:
            features = features / features.norm(dim=1, keepdim=True)

        return features.to(torch.float32)
