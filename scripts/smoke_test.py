#!/usr/bin/env python
"""Verify the environment and optionally run an end-to-end MedSigLIP inference."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import tempfile
from pathlib import Path


REQUIRED_MODULES = ("torch", "transformers", "PIL", "lightning", "medsiglip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--run-model", action="store_true")
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

    if not args.run_model:
        print("Environment smoke test passed. Add --run-model for gated-model inference.")
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
