"""PyTorch Lightning classifier built on the MedSigLIP vision encoder."""

from __future__ import annotations

from collections.abc import Sequence

import lightning as L
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from transformers import SiglipVisionModel


class MedSigLIPClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "google/medsiglip-448",
        class_names: Sequence[str] = (),
        freeze_encoder: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("class_names must not be empty")
        self.save_hyperparameters()
        self.class_names = list(class_names)
        self.freeze_encoder = freeze_encoder
        self.vision_encoder = SiglipVisionModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.vision_encoder.config.hidden_size, len(self.class_names))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.validation_outputs: list[dict[str, torch.Tensor]] = []
        self.test_outputs: list[dict[str, torch.Tensor]] = []
        self._set_encoder_trainability()

    def _set_encoder_trainability(self) -> None:
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = not self.freeze_encoder
        if self.freeze_encoder:
            self.vision_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.vision_encoder.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(pixel_values=pixel_values).pooler_output
        return self.classifier(features)

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch["label"].float()
        logits = self(batch["pixel_values"])
        return self.loss_fn(logits, labels), logits, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(batch["label"]))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch)
        self.validation_outputs.append({"logits": logits.detach(), "labels": labels.detach()})
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(labels))

    def on_validation_epoch_end(self) -> None:
        self._log_auc(self.validation_outputs, "val")
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch)
        self.test_outputs.append({"logits": logits.detach(), "labels": labels.detach()})
        self.log("test_loss", loss, on_epoch=True, batch_size=len(labels))

    def on_test_epoch_start(self) -> None:
        self.test_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._log_auc(self.test_outputs, "test")

    def prediction_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.test_outputs:
            raise RuntimeError("No test predictions are available")
        logits = torch.cat([item["logits"] for item in self.test_outputs])
        labels = torch.cat([item["labels"] for item in self.test_outputs])
        return torch.sigmoid(logits).cpu().numpy(), labels.cpu().numpy()

    def _log_auc(self, outputs: list[dict[str, torch.Tensor]], prefix: str) -> None:
        if not outputs:
            return
        logits = torch.cat([item["logits"] for item in outputs]).cpu()
        labels = torch.cat([item["labels"] for item in outputs]).cpu().numpy()
        probabilities = torch.sigmoid(logits).numpy()
        aucs = []
        for index, name in enumerate(self.class_names):
            if np.unique(labels[:, index]).size < 2:
                continue
            auc = roc_auc_score(labels[:, index], probabilities[:, index])
            aucs.append(auc)
            self.log(f"{prefix}_auc_{name}", auc, prog_bar=False)
        if aucs:
            self.log(f"{prefix}_macro_auc", float(np.mean(aucs)), prog_bar=True)

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

