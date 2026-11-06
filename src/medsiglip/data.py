"""Data loading and modality-aware image preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor


def prepare_medical_image(image: Image.Image) -> Image.Image:
    """Convert an image to 8-bit RGB without distorting existing RGB colors.

    Grayscale and high-bit-depth medical images are min-max scaled to 8-bit and
    replicated across three channels. Existing color images are converted to RGB
    directly, preserving their relative channel values.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = array.astype(np.float32)
        low, high = float(array.min()), float(array.max())
        if high > low:
            array = ((array - low) * 255.0 / (high - low)).astype(np.uint8)
        else:
            array = np.zeros_like(array, dtype=np.uint8)
        array = np.repeat(array[..., None], 3, axis=-1)
        return Image.fromarray(array, mode="RGB")
    return image.convert("RGB")


class MedicalImageDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str | Path],
        labels: np.ndarray,
        processor,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self.image_paths = [Path(path) for path in image_paths]
        self.labels = np.asarray(labels, dtype=np.float32)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.image_paths[index]
        with Image.open(path) as source:
            image = prepare_medical_image(source)
        encoded = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "label": torch.from_numpy(self.labels[index]),
            "path": str(path),
        }


def build_dataloaders(
    frame,
    image_root: str | Path,
    class_names: Sequence[str],
    model_name: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    processor = AutoProcessor.from_pretrained(model_name)
    root = Path(image_root)

    def make(split: str, shuffle: bool) -> DataLoader:
        subset = frame.loc[frame["Split"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"Split {split!r} is empty")
        paths = [root / path for path in subset["ImagePath"].astype(str)]
        dataset = MedicalImageDataset(paths, subset[list(class_names)].to_numpy(), processor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    return make("Train", True), make("Valid", False), make("Test", False)

