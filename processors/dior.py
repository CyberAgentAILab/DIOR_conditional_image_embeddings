import os
import time
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from entities.config import EmbeddingSource, InferenceConfig
from entities.prompt import get_prompt

from .base import BaseEmbeddingProcessor


class DIOREmbeddingProcessor(BaseEmbeddingProcessor):
    """Embedding extraction for Vision Language Models with KV cache support."""

    def __init__(self, config: InferenceConfig):
        super().__init__(config)

        self.has_chat_template = hasattr(self.processor, "tokenizer") and getattr(
            self.processor.tokenizer, "chat_template", None
        )

        self._kv_cache = None
        self._kv_cache_length = None
        self._cached_inputs = None

    def run(self) -> None:
        image_files = self._get_image_files()

        start_time = time.time()

        if self.config.use_cache or self.config.batch_size == 1:
            self._run_single_mode(image_files)
        else:
            self._run_batch_mode(image_files)

        elapsed = time.time() - start_time
        mode = "KV cache" if self.config.use_cache else f"batch={self.config.batch_size}"
        print(f"Processed {self.aspects} ({mode}) in {elapsed:.2f}s")

    def _run_single_mode(self, image_files: List[str]) -> None:
        for path in tqdm(image_files):
            if self.config.use_cache:
                self._process_with_cache(path)
            else:
                self._process_single_image(path)

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
            self._extract_and_save(image, prompt, output_path)

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
        prompts = [prompt] * len(to_process)

        self._extract_and_save_batch(images_to_process, prompts, paths_to_process)
        torch.cuda.empty_cache()

    def _extract_and_save(self, image: Image.Image, prompt: str, output_path: str) -> None:
        inputs = self._prepare_single_input(image, prompt)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        self._save_layers(outputs.hidden_states, output_path, prompt, inputs.get("attention_mask"))

    def _extract_and_save_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        output_paths: List[str],
    ) -> None:
        inputs = self._prepare_batch_input(images, prompts)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        self._save_layers_batch(outputs.hidden_states, inputs["attention_mask"], output_paths, prompts)

    def _save_layers(
        self,
        hidden_states: tuple,
        output_path: str,
        prompt: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        base = os.path.dirname(output_path)
        stem = os.path.splitext(os.path.basename(output_path))[0]

        layer_indices = range(len(hidden_states)) if self.config.save_all else [self.config.num_layer]

        for layer_idx in layer_indices:
            hidden = hidden_states[layer_idx]
            embedding = self._extract_token_embedding(hidden, attention_mask)

            out_dir = base + f"-num_layer={layer_idx}-num_token={self.config.num_token}"
            out_dir = self._append_source_suffix(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            data = {
                "embedding": embedding.numpy().tolist(),
                "prompt": prompt,
                "source": self.config.embedding_source,
            }
            if self.config.save_all:
                data["layer"] = layer_idx

            self._save_json(os.path.join(out_dir, stem + ".json"), data)

    def _save_layers_batch(
        self,
        hidden_states: tuple,
        attention_mask: torch.Tensor,
        output_paths: List[str],
        prompts: List[str],
    ) -> None:
        stems = [os.path.splitext(os.path.basename(p))[0] for p in output_paths]
        layer_indices = range(len(hidden_states)) if self.config.save_all else [self.config.num_layer]

        for layer_idx in layer_indices:
            hidden = hidden_states[layer_idx]
            embeddings = self._extract_token_embedding(hidden, attention_mask)
            embeddings = embeddings.detach().cpu().to(torch.float32)

            base_dirs = [
                self._append_source_suffix(
                    os.path.dirname(p) + f"-num_layer={layer_idx}-num_token={self.config.num_token}"
                )
                for p in output_paths
            ]
            for bd in set(base_dirs):
                os.makedirs(bd, exist_ok=True)

            for i in range(len(output_paths)):
                data = {
                    "embedding": embeddings[i].numpy().tolist(),
                    "prompt": prompts[i],
                    "source": self.config.embedding_source,
                }
                if self.config.save_all:
                    data["layer"] = layer_idx

                self._save_json(os.path.join(base_dirs[i], stems[i] + ".json"), data)

    def _extract_token_embedding(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, S, D = hidden.shape

        if self.config.num_token == -1 and attention_mask is not None:
            last_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            last_idx = torch.clamp(last_idx, min=0, max=S - 1)
            idx = last_idx.view(B, 1, 1).expand(-1, 1, D)
            return torch.gather(hidden, dim=1, index=idx).squeeze(1).detach().cpu().to(torch.float32)
        elif self.config.num_token == -1:
            return hidden[:, -1, :].detach().cpu().to(torch.float32)
        else:
            return hidden[:, self.config.num_token, :].detach().cpu().to(torch.float32)

    def _process_with_cache(self, image_path: str) -> None:
        image = self._load_image(image_path)
        if image is None:
            return

        filename = self._get_json_filename(image_path)

        first_aspect, first_prompt = self._find_first_cacheable_aspect()
        if first_aspect is None:
            for aspect in self.aspects:
                self._process_aspect(image, aspect, filename)
            return

        try:
            first_outputs = self._compute_kv_cache(image, first_prompt)
        except Exception as e:
            print(f"KV cache failed: {e}")
            for aspect in self.aspects:
                self._process_aspect(image, aspect, filename)
            return

        for aspect in self.aspects:
            output_path = self._build_output_path(filename, aspect)
            if self._output_exists(output_path):
                continue

            prompt = get_prompt(self.config.prompt_type, aspect)

            if aspect is None:
                self._process_aspect(image, aspect, filename)
            elif aspect == first_aspect:
                self._save_layers(first_outputs.hidden_states, output_path, prompt)
            else:
                self._process_with_cached_kv(image, prompt, output_path, filename, aspect)

        self._clear_kv_cache()

    def _process_aspect(self, image: Image.Image, aspect: Optional[str], filename: str) -> None:
        output_path = self._build_output_path(filename, aspect)
        if self._output_exists(output_path):
            return

        prompt = get_prompt(self.config.prompt_type, aspect)
        self._extract_and_save(image, prompt, output_path)

    def _process_with_cached_kv(
        self,
        image: Image.Image,
        prompt: str,
        output_path: str,
        filename: str,
        aspect: Optional[str],
    ) -> None:
        try:
            new_inputs = self._prepare_single_input(image, prompt)
            prefix_len = self._find_common_prefix_length(self._cached_inputs, new_inputs)

            outputs, suffix_len = self._forward_with_kv_cache(image, prompt, prefix_len)

            if outputs is None or suffix_len == 0:
                self._process_aspect(image, aspect, filename)
            else:
                self._save_layers(outputs.hidden_states, output_path, prompt)
        except Exception as e:
            print(f"KV forward failed: {e}")
            self._process_aspect(image, aspect, filename)

    def _find_first_cacheable_aspect(self) -> tuple:
        for aspect in self.aspects:
            if aspect is not None:
                return aspect, get_prompt(self.config.prompt_type, aspect)
        return None, None

    def _compute_kv_cache(self, image: Image.Image, prompt: str):
        inputs = self._prepare_single_input(image, prompt)

        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True, output_hidden_states=True)

        self._kv_cache = outputs.past_key_values
        self._kv_cache_length = self._get_kv_cache_length(outputs.past_key_values)
        self._cached_inputs = inputs

        return outputs

    def _forward_with_kv_cache(self, image: Image.Image, prompt: str, common_prefix_len: int):
        device, _ = self._get_device_and_dtype()

        new_inputs = self._prepare_single_input(image, prompt)
        suffix_ids = new_inputs["input_ids"][:, common_prefix_len:]
        suffix_len = suffix_ids.shape[1]

        if suffix_len == 0:
            return None, 0

        cached_ids_len = self._cached_inputs["input_ids"].shape[1]
        expansion = self._kv_cache_length - cached_ids_len
        kv_len = common_prefix_len + expansion

        truncated_kv = self._truncate_kv_cache(self._kv_cache, kv_len)

        attn = torch.ones((1, kv_len + suffix_len), dtype=torch.long, device=device)
        pos = torch.arange(kv_len, kv_len + suffix_len, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                input_ids=suffix_ids,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=truncated_kv,
                use_cache=False,
                output_hidden_states=True,
            )

        return outputs, suffix_len

    def _clear_kv_cache(self) -> None:
        self._kv_cache = None
        self._kv_cache_length = None
        self._cached_inputs = None
        torch.cuda.empty_cache()

    def _get_kv_cache_length(self, kv_cache) -> int:
        if kv_cache is None:
            return 0

        first = kv_cache[0]
        if isinstance(first, tuple) and len(first) >= 2:
            key = first[0]
            return key.shape[2] if key.dim() == 4 else key.shape[1]

        if hasattr(kv_cache, "get_seq_length"):
            return kv_cache.get_seq_length()

        return 0

    def _truncate_kv_cache(self, kv_cache, length: int):
        from transformers.cache_utils import DynamicCache

        if isinstance(kv_cache, DynamicCache):
            truncated = DynamicCache()
            for i in range(len(kv_cache)):
                k, v = kv_cache[i]
                truncated.update(k[:, :, :length, :].clone(), v[:, :, :length, :].clone(), i)
            return truncated

        return tuple((kv[0][:, :, :length, :].clone(), kv[1][:, :, :length, :].clone()) for kv in kv_cache)

    def _find_common_prefix_length(self, inputs_a: dict, inputs_b: dict) -> int:
        ids_a = inputs_a["input_ids"][0].tolist()
        ids_b = inputs_b["input_ids"][0].tolist()

        length = 0
        for i in range(min(len(ids_a), len(ids_b))):
            if ids_a[i] == ids_b[i]:
                length = i + 1
            else:
                break
        return length

    def _prepare_single_input(self, image: Image.Image, prompt: str) -> dict:
        if self.has_chat_template:
            conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=f"<image>{prompt}", images=image, return_tensors="pt", padding=True)

        inputs = self._to_device(inputs)

        if "attention_mask" in inputs and "position_ids" not in inputs:
            attn = inputs["attention_mask"]
            pos = attn.long().cumsum(dim=1) - 1
            pos.masked_fill_(attn == 0, 0)
            inputs["position_ids"] = pos

        return inputs

    def _prepare_batch_input(self, images: List[Image.Image], prompts: List[str]) -> dict:
        if self.has_chat_template:
            texts = [
                self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
        else:
            texts = [f"<image>{p}" for p in prompts]

        inputs = self.processor(
            text=texts,
            images=[[img] for img in images],
            return_tensors="pt",
            padding=True,
        )
        return self._to_device(inputs)

    def _build_output_path(self, filename: str, aspect: Optional[str]) -> str:
        out_dir = self._get_base_output_dir()

        aspect_str = aspect.replace(" ", "_") if aspect else str(aspect)
        out_dir += f"-{self.config.prompt_type}-{aspect_str}"

        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, filename)

    def _append_source_suffix(self, base: str) -> str:
        suffix = f"-embedsrc={self.config.embedding_source}"
        if self.config.embedding_source in (EmbeddingSource.FIRST_OUTPUT, EmbeddingSource.MEAN_OUTPUT):
            suffix += f"-gen_max_new_tokens={self.config.gen_max_new_tokens}"
            if self.config.gen_do_sample:
                suffix += f"-sample-temp={self.config.gen_temperature:.2f}"
        return base + suffix

    def _output_exists(self, output_path: str) -> bool:
        base = os.path.dirname(output_path)
        stem = os.path.splitext(os.path.basename(output_path))[0]
        check_dir = base + f"-num_layer={self.config.num_layer}-num_token={self.config.num_token}"
        check_dir = self._append_source_suffix(check_dir)
        return os.path.exists(os.path.join(check_dir, stem + ".json"))
