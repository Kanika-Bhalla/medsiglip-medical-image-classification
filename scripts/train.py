#!/usr/bin/env python
"""Train a frozen-encoder or fully fine-tuned MedSigLIP classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from medsiglip.data import build_dataloaders
from medsiglip.model import MedSigLIPClassifier


DEFAULT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Lung Opacity",
    "Pneumothorax",
    "Pleural Effusion",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="google/medsiglip-448")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5656)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    frame = pd.read_csv(args.csv)
    train_loader, valid_loader, _ = build_dataloaders(
        frame,
        args.image_root,
        args.classes,
        args.model_name,
        args.batch_size,
        args.num_workers,
    )
    model = MedSigLIPClassifier(
        model_name=args.model_name,
        class_names=args.classes,
        freeze_encoder=args.freeze_encoder,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        deterministic=True,
        gradient_clip_val=1.0,
        callbacks=[checkpoint, EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        logger=CSVLogger(output, name="logs"),
    )
    trainer.fit(model, train_loader, valid_loader)
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()

