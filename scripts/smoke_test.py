#!/usr/bin/env python
"""Verify the environment and optionally run an end-to-end MedSigLIP inference."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import tempfile
from pathlib import Path


REQUIRED_MODULES = ("torch", "transformers", "peft", "PIL", "lightning", "medsiglip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument(
        "--check-lora",
        action="store_true",
        help="Download the gated vision encoder and verify that LoRA attaches correctly",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    print(f"Python: {platform.python_version()}")
    for module_name in REQUIRED_MODULES:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "installed")
        print(f"{module_name}: {version}")

    import torch

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA is required for this smoke test but is not available")

    if args.check_lora:
        from medsiglip.model import MedSigLIPClassifier

        lora_model = MedSigLIPClassifier(
            class_names=["cardiomegaly", "pleural effusion"],
            adaptation_mode="lora",
        )
        if args.device == "cuda" or (args.device == "auto" and cuda_available):
            lora_model = lora_model.cuda()
        summary = lora_model.parameter_summary()
        print(
            "LoRA attachment passed: "
            f"{summary['trainable']:,} trainable / {summary['total']:,} total "
            f"({summary['trainable_percent']:.3f}%)"
        )

    if not args.run_model:
        suffix = "" if args.check_lora else " Add --run-model for gated-model inference."
        print(f"Environment smoke test passed.{suffix}")
        return

    from create_demo_image import create_synthetic_cxr
    from zero_shot import run_zero_shot

    with tempfile.TemporaryDirectory() as temporary_directory:
        image_path = create_synthetic_cxr(Path(temporary_directory) / "synthetic_cxr.png")
        result = run_zero_shot(
            image_path=image_path,
            labels=["cardiomegaly", "pleural effusion"],
            model_name="google/medsiglip-448",
            device_choice=args.device,
        )
    print(json.dumps(result, indent=2))
    print("End-to-end model smoke test passed. Synthetic-image scores are not clinical results.")


if __name__ == "__main__":
    main()
