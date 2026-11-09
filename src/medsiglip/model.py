"""PyTorch Lightning classifier built on the MedSigLIP vision encoder."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
        adaptation_mode: str | None = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("class_names must not be empty")
        resolved_mode = adaptation_mode or ("frozen" if freeze_encoder else "full")
        if resolved_mode not in {"frozen", "lora", "full"}:
            raise ValueError("adaptation_mode must be one of: frozen, lora, full")
        if lora_r <= 0 or lora_alpha <= 0:
            raise ValueError("lora_r and lora_alpha must be positive")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not lora_target_modules:
            raise ValueError("lora_target_modules must not be empty")

        self.save_hyperparameters(
            {
                "model_name": model_name,
                "class_names": list(class_names),
                "freeze_encoder": resolved_mode == "frozen",
                "adaptation_mode": resolved_mode,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": list(lora_target_modules),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }
        )
        self.class_names = list(class_names)
        self.adaptation_mode = resolved_mode
        self.freeze_encoder = resolved_mode == "frozen"

        base_encoder = SiglipVisionModel.from_pretrained(model_name)
        hidden_size = base_encoder.config.hidden_size
        if resolved_mode == "lora":
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=list(lora_target_modules),
            )
            self.vision_encoder = get_peft_model(base_encoder, lora_config)
        else:
            self.vision_encoder = base_encoder
        self.classifier = nn.Linear(hidden_size, len(self.class_names))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.validation_outputs: list[dict[str, torch.Tensor]] = []
        self.test_outputs: list[dict[str, torch.Tensor]] = []
        self._set_encoder_trainability()

    def _set_encoder_trainability(self) -> None:
        if self.adaptation_mode == "frozen":
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False
            self.vision_encoder.eval()
        elif self.adaptation_mode == "full":
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = True
        # PEFT already freezes the base weights and leaves only LoRA parameters trainable.

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.vision_encoder.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(pixel_values=pixel_values).pooler_output
        return self.classifier(features)

    def parameter_summary(self) -> dict[str, float | int]:
        """Return total and trainable parameter counts for experiment reporting."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "trainable_percent": 100.0 * trainable / total,
        }

    def export_lora_bundle(self, output_dir: str | Path) -> Path:
        """Export the compact vision adapter plus the supervised classifier head."""
        if self.adaptation_mode != "lora":
            raise RuntimeError("LoRA bundle export is only available in lora mode")
        output = Path(output_dir)
        adapter_directory = output / "vision_adapter"
        adapter_directory.mkdir(parents=True, exist_ok=True)
        self.vision_encoder.save_pretrained(adapter_directory)
        torch.save(
            {
                "class_names": self.class_names,
                "classifier_state_dict": self.classifier.state_dict(),
            },
            output / "classifier_head.pt",
        )
        return output

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

