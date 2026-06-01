import os
from typing import List, Optional

import torch
from PIL import Image

from entities.config import InferenceConfig, TextEncoder
from entities.prompt import get_prompt

from .base import BaseEmbeddingProcessor


class GenerativeEmbeddingProcessor(BaseEmbeddingProcessor):
    """Generate text with VLM, then encode with Sentence-T5 for embeddings."""

    def __init__(self, config: InferenceConfig, text_encoder_id: Optional[str] = None):
        super().__init__(config)

        self.has_chat_template = hasattr(self.processor, "tokenizer") and getattr(
            self.processor.tokenizer, "chat_template", None
        )

        self.text_encoder_id = text_encoder_id or TextEncoder.SENTENCE_T5_BASE.value
        self._load_text_encoder(self.text_encoder_id)

    def _load_text_encoder(self, model_id: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.text_encoder = SentenceTransformer(model_id)
        self.text_encoder.eval()

    def _process_single_image(self, image_path: str) -> None:
        image = self._load_image(image_path)
        if image is None:
            return

        filename = self._get_json_filename(image_path)

        for aspect in self.aspects:
            output_path = self._build_output_path(filename, aspect)
            if self._output_exists(output_path):
                continue

            prompt = get_prompt(self.config.prompt_type, aspect)
            self._generate_and_encode(image, prompt, output_path)

        torch.cuda.empty_cache()

    def _process_batch(self, image_paths: List[str], aspect: Optional[str]) -> None:
        images, valid_paths = self._load_images(image_paths)
        if not images:
            return

        prompt = get_prompt(self.config.prompt_type, aspect)
        filenames = [self._get_json_filename(p) for p in valid_paths]
        output_paths = [self._build_output_path(f, aspect) for f in filenames]

        to_process = [(img, path) for img, path in zip(images, output_paths) if not self._output_exists(path)]

        if not to_process:
            return

        images_to_process = [x[0] for x in to_process]
        paths_to_process = [x[1] for x in to_process]

        for img, path in zip(images_to_process, paths_to_process):
            self._generate_and_encode(img, prompt, path)

        torch.cuda.empty_cache()

    def _generate_and_encode(self, image: Image.Image, prompt: str, output_path: str) -> None:
        generated_text = self._generate_text(image, prompt)
        embedding = self._encode_text(generated_text)
        self._save_embedding(embedding, output_path, prompt, generated_text)

    def _generate_text(self, image: Image.Image, prompt: str) -> str:
        inputs = self._prepare_input(image, prompt)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.gen_max_new_tokens,
                do_sample=self.config.gen_do_sample,
                temperature=self.config.gen_temperature if self.config.gen_do_sample else None,
            )

        generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

    def _encode_text(self, text: str) -> List[float]:
        with torch.no_grad():
            embedding = self.text_encoder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _prepare_input(self, image: Image.Image, prompt: str) -> dict:
        if self.has_chat_template:
            conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = self.processor(text=text, images=image, return_tensors="pt")
        else:
            inputs = self.processor(text=f"<image>{prompt}", images=image, return_tensors="pt")

        return self._to_device(inputs)

    def _save_embedding(
        self,
        embedding: List[float],
        output_path: str,
        prompt: str,
        generated_text: str,
    ) -> None:
        base = os.path.dirname(output_path)
        stem = os.path.splitext(os.path.basename(output_path))[0]

        out_dir = base + f"-embedsrc=generative-{self.text_encoder_id.replace('/', '_')}"
        out_dir += f"-max_tokens={self.config.gen_max_new_tokens}"
        if self.config.gen_do_sample:
            out_dir += f"-sample-temp={self.config.gen_temperature:.2f}"

        os.makedirs(out_dir, exist_ok=True)

        self._save_json(
            os.path.join(out_dir, stem + ".json"),
            {
                "embedding": embedding,
                "prompt": prompt,
                "generated_text": generated_text,
                "source": "generative",
                "text_encoder": self.text_encoder_id,
            },
        )

    def _build_output_path(self, filename: str, aspect: Optional[str]) -> str:
        out_dir = self._get_base_output_dir()

        aspect_str = aspect.replace(" ", "_") if aspect else str(aspect)
        out_dir += f"-{self.config.prompt_type}-{aspect_str}"

        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, filename)

    def _output_exists(self, output_path: str) -> bool:
        base = os.path.dirname(output_path)
        stem = os.path.splitext(os.path.basename(output_path))[0]

        check_dir = base + f"-embedsrc=generative-{self.text_encoder_id.replace('/', '_')}"
        check_dir += f"-max_tokens={self.config.gen_max_new_tokens}"
        if self.config.gen_do_sample:
            check_dir += f"-sample-temp={self.config.gen_temperature:.2f}"

        return os.path.exists(os.path.join(check_dir, stem + ".json"))
