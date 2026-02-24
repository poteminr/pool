from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class SigLIP2EmbeddingBackend:
    name = "siglip2"

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-naflex",
        max_num_patches: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._max_num_patches = max(64, int(max_num_patches))
        self._text_max_length = 64

        if torch.cuda.is_available():
            preferred_device = "cuda:0"
        elif torch.backends.mps.is_available():
            preferred_device = "mps"
        else:
            preferred_device = "cpu"

        dtype = torch.float16 if preferred_device != "cpu" else torch.float32
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        try:
            model = model.to(preferred_device)
            self._device = preferred_device
        except Exception:
            model = model.to("cpu")
            self._device = "cpu"
        model.eval()

        self._model = model
        self._processor = processor

        projection_dim = getattr(getattr(self._model, "config", None), "projection_dim", None)
        hidden_size = getattr(getattr(self._model, "config", None), "hidden_size", None)
        if isinstance(projection_dim, int) and projection_dim > 0:
            dimension = projection_dim
        elif isinstance(hidden_size, int) and hidden_size > 0:
            dimension = hidden_size
        else:
            dimension = 768
        self._dimension = int(dimension)
        self.name = f"siglip2::{model_name}::max_patches={self._max_num_patches}"

    def _process_images(self, images: list[Image.Image]):
        try:
            return self._processor(
                images=images,
                max_num_patches=self._max_num_patches,
                return_tensors="pt",
            )
        except TypeError:
            return self._processor(images=images, return_tensors="pt")

    @staticmethod
    def _normalize_label_text(text: str) -> str:
        clean = " ".join(text.strip().split()).lower()
        return clean or " "

    def _process_text(self, texts: list[str]):
        normalized = [self._normalize_label_text(text) for text in texts]
        kwargs = {
            "text": normalized,
            "padding": "max_length",
            "truncation": True,
            "max_length": self._text_max_length,
            "return_tensors": "pt",
        }
        try:
            return self._processor(**kwargs)
        except TypeError:
            kwargs.pop("max_length", None)
            return self._processor(**kwargs)

    def embed(self, paths: list[Path], batch_size: int) -> list[list[float]]:
        torch = self._torch
        vectors: list[list[float]] = []
        step = max(1, int(batch_size))
        for start in range(0, len(paths), step):
            batch_paths = paths[start : start + step]
            images: list[Image.Image] = []
            valid: list[bool] = []
            for path in batch_paths:
                try:
                    with Image.open(path) as image:
                        images.append(ImageOps.exif_transpose(image.convert("RGB")))
                        valid.append(True)
                except Exception:
                    valid.append(False)
            if not images:
                vectors.extend([np.zeros(self._dimension, dtype=np.float32).tolist() for _ in batch_paths])
                continue

            inputs = self._process_images(images=images)
            inputs = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                if hasattr(self._model, "get_image_features"):
                    image_features = self._model.get_image_features(**inputs)
                else:
                    outputs = self._model(**inputs)
                    image_features = getattr(outputs, "image_embeds", None)
                    if image_features is None:
                        image_features = getattr(outputs, "last_hidden_state", None)
                    if image_features is None:
                        raise RuntimeError("SigLIP2 backend: could not extract image features")
                    if image_features.ndim == 3:
                        image_features = image_features[:, 0, :]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                encoded = image_features.detach().cpu().numpy().astype(np.float32)

            idx = 0
            for ok in valid:
                if ok:
                    vectors.append(encoded[idx].tolist())
                    idx += 1
                else:
                    vectors.append(np.zeros(encoded.shape[1], dtype=np.float32).tolist())
        return vectors

    def embed_text(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []

        torch = self._torch
        vectors: list[list[float]] = []
        step = max(1, int(batch_size))
        for start in range(0, len(texts), step):
            chunk = texts[start : start + step]
            payload = self._process_text(texts=chunk)
            payload = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in payload.items()
            }

            with torch.no_grad():
                if hasattr(self._model, "get_text_features"):
                    text_features = self._model.get_text_features(**payload)
                else:
                    outputs = self._model(**payload)
                    text_features = getattr(outputs, "text_embeds", None)
                    if text_features is None:
                        text_features = getattr(outputs, "last_hidden_state", None)
                    if text_features is None:
                        raise RuntimeError("SigLIP2 backend: could not extract text features")
                    if text_features.ndim == 3:
                        text_features = text_features[:, 0, :]
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                encoded = text_features.detach().cpu().numpy().astype(np.float32)
            vectors.extend(encoded.tolist())

        return vectors


def build_siglip2_backend(siglip2_max_patches: int, log: Callable[[str], None]) -> SigLIP2EmbeddingBackend:
    backend = SigLIP2EmbeddingBackend(max_num_patches=siglip2_max_patches)
    log(f"Embedding backend: {backend.name}")
    return backend
