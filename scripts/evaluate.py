#!/usr/bin/env python
"""Select thresholds on validation data and evaluate the held-out test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import pandas as pd

from medsiglip.data import build_dataloaders
from medsiglip.evaluation import evaluate_classes
from medsiglip.model import MedSigLIPClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def infer(trainer, model, loader):
    trainer.test(model, dataloaders=loader, verbose=False)
    return model.prediction_arrays()


def main() -> None:
    args = parse_args()
    model = MedSigLIPClassifier.load_from_checkpoint(args.checkpoint)
    frame = pd.read_csv(args.csv)
    _, valid_loader, test_loader = build_dataloaders(
        frame,
        args.image_root,
        model.class_names,
        model.hparams.model_name,
        args.batch_size,
        args.num_workers,
    )
    trainer = L.Trainer(accelerator="auto", devices=1, logger=False, enable_checkpointing=False)
    validation_probabilities, validation_labels = infer(trainer, model, valid_loader)
    test_probabilities, test_labels = infer(trainer, model, test_loader)
    results = evaluate_classes(
        validation_labels,
        validation_probabilities,
        test_labels,
        test_probabilities,
        model.class_names,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

