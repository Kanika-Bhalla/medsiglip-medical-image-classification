"""Prompt construction utilities for MedSigLIP zero-shot classification."""

from __future__ import annotations

from collections.abc import Sequence


DEFAULT_ZERO_SHOT_LABELS = (
    "cardiomegaly",
    "pleural effusion",
    "pneumothorax",
    "pulmonary edema",
    "airspace opacity",
)


def build_binary_prompts(labels: Sequence[str]) -> list[str]:
    """Return positive/negative chest-X-ray prompts for every finding.

    Prompts are interleaved as positive, negative pairs so that logits can be
    reshaped to ``[n_labels, 2]`` and normalized within each finding.
    """
    if not labels:
        raise ValueError("At least one candidate label is required")
    prompts: list[str] = []
    for raw_label in labels:
        label = raw_label.strip().lower()
        if not label:
            raise ValueError("Candidate labels must not be blank")
        prompts.extend(
            [
                f"a chest x-ray showing {label}",
                f"a chest x-ray with no evidence of {label}",
            ]
        )
    return prompts
