#!/usr/bin/env python
"""Run paired-prompt MedSigLIP zero-shot classification on one medical image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from medsiglip.data import prepare_medical_image
from medsiglip.prompts import DEFAULT_ZERO_SHOT_LABELS, build_binary_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Local PNG/JPEG/TIFF image")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_ZERO_SHOT_LABELS))
    parser.add_argument("--model-name", default="google/medsiglip-448")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def run_zero_shot(
    image_path: str | Path,
    labels: list[str],
    model_name: str,
    device_choice: str = "auto",
) -> dict[str, object]:
    """Return independent paired-prompt probabilities for each finding."""
    import torch
    from transformers import AutoModel, AutoProcessor

    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = (
        "cuda"
        if device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    prompts = build_binary_prompts(labels)

    with Image.open(image_path) as source:
        image = prepare_medical_image(source)
    inputs = processor(
        text=prompts,
        images=image,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        paired_logits = outputs.logits_per_image[0].reshape(len(labels), 2)
        positive_probabilities = torch.softmax(paired_logits, dim=1)[:, 0]

    findings = [
        {"label": label, "probability": float(probability)}
        for label, probability in zip(labels, positive_probabilities.cpu(), strict=True)
    ]
    findings.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "model": model_name,
        "image": str(image_path),
        "scoring": "positive-vs-negative paired prompts",
        "findings": findings,
    }


def main() -> None:
    args = parse_args()
    result = run_zero_shot(args.image, args.labels, args.model_name, args.device)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
