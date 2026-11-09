#!/usr/bin/env python
"""Create a synthetic chest-like image for non-clinical pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def create_synthetic_cxr(output_path: str | Path, size: int = 448) -> Path:
    """Write a deterministic, synthetic grayscale image with a chest-like layout."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (size, size), color=28)
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 28, size - 60, size - 24), fill=118)
    draw.ellipse((92, 72, size // 2 - 10, size - 82), fill=45)
    draw.ellipse((size // 2 + 10, 72, size - 92, size - 82), fill=45)
    draw.rectangle((size // 2 - 7, 42, size // 2 + 7, size - 58), fill=160)
    draw.ellipse((size // 2 - 36, size // 2 - 8, size // 2 + 76, size - 90), fill=92)
    image = image.filter(ImageFilter.GaussianBlur(radius=3.0))
    image.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/synthetic_cxr.png")
    parser.add_argument("--size", type=int, default=448)
    args = parser.parse_args()
    path = create_synthetic_cxr(args.output, args.size)
    print(f"Created synthetic, non-clinical smoke-test image: {path}")


if __name__ == "__main__":
    main()
