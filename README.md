# MedSigLIP Medical Image Classification

Leakage-aware transfer learning and zero-shot medical image classification with
[MedSigLIP](https://huggingface.co/google/medsiglip-448).

This portfolio project demonstrates a complete medical vision-language workflow:

- zero-shot image-text matching;
- extraction of pretrained medical image representations;
- frozen-encoder linear probing;
- full encoder fine-tuning;
- validation-set threshold selection with Youden's index; and
- independent test-set ROC-AUC, sensitivity, and specificity evaluation.

## Why this project matters

Medical datasets often contain several images from the same patient or study.
Randomly splitting image rows can leak related images across train, validation,
and test sets. This implementation provides a group-aware split utility so that
all images from the same patient (preferred) or study remain in one partition.

## Pipeline

```mermaid
flowchart LR
    A[Medical images] --> B[Modality-aware preprocessing]
    B --> C[MedSigLIP vision encoder]
    C --> D[Linear classification head]
    D --> E[Validation thresholds]
    E --> F[Independent test evaluation]
```

## Preliminary exercise results

The original 1,000-image EmoryCXR training exercise compared a frozen vision
encoder with full fine-tuning across five findings. The saved notebook reported:

| Approach | Macro ROC-AUC |
|---|---:|
| Frozen encoder / linear probe | 0.8124 |
| Full fine-tuning | 0.7921 |

![Preliminary comparison](docs/preliminary_results.svg)

These values are retained for transparent documentation, but they are **not
presented as final validated performance**. The original exercise assigned
individual image rows to 600/200/200 splits. A patient- or study-grouped rerun
is required before making a definitive performance claim.

## Repository structure

```text
.
├── src/medsiglip/
│   ├── data.py              # Image preprocessing and PyTorch Dataset
│   ├── evaluation.py        # ROC-AUC and Youden-threshold evaluation
│   └── model.py             # PyTorch Lightning MedSigLIP classifier
├── scripts/
│   ├── split_dataset.py     # Group-aware 60/20/20 split generation
│   ├── train.py             # Frozen-encoder or full fine-tuning
│   └── evaluate.py          # Validation thresholding + test evaluation
├── results/
│   └── preliminary_metrics.csv
└── docs/
    └── preliminary_results.svg
```

## Dataset manifest

The scripts expect a CSV containing:

- `ImagePath`: relative or absolute image path;
- a patient or study identifier such as `empi_anon` or
  `AccessionNumber_anon`;
- `Pathology`: one of the target classes; and
- after splitting, `Split`: `Train`, `Valid`, or `Test`.

No EmoryCXR images, identifiers, clinical data, or model checkpoints are
distributed in this repository.

## Leakage-safe splitting

```bash
python scripts/split_dataset.py \
  --input metadata.csv \
  --output metadata_grouped_split.csv \
  --group-col empi_anon \
  --label-col Pathology
```

If a patient identifier is unavailable, use `AccessionNumber_anon` to prevent
study-level leakage.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MedSigLIP is gated on Hugging Face. Authenticate outside the source code:

```bash
huggingface-cli login
```

Never commit access tokens, API keys, `.env` files, patient data, or model
checkpoints.

## Training

Frozen encoder / linear probe:

```bash
python scripts/train.py \
  --csv metadata_grouped_split.csv \
  --image-root /path/to/images \
  --output-dir outputs/linear_probe \
  --freeze-encoder
```

Full fine-tuning:

```bash
python scripts/train.py \
  --csv metadata_grouped_split.csv \
  --image-root /path/to/images \
  --output-dir outputs/full_finetuning \
  --learning-rate 1e-5
```

## Evaluation

```bash
python scripts/evaluate.py \
  --csv metadata_grouped_split.csv \
  --image-root /path/to/images \
  --checkpoint outputs/linear_probe/best.ckpt \
  --output results/grouped_test_metrics.json
```

Thresholds are selected on the validation split and applied once to the test
split. The test split is never used to choose thresholds or hyperparameters.

## Attribution

Portfolio implementation and pipeline modifications: **Kanika Bhalla, PhD**.

This work builds on:

- Google's MedSigLIP model and public quick-start materials;
- the Emory Health AI Datathon 2025 instructional workflow developed by Frank
  Li and collaborators at the HITI Lab; and
- the public SCIN dataset for the original dermatology demonstration.

The code in this repository is a cleaned portfolio implementation. It is not an
official Google, Emory University, or HITI Lab repository.

## Responsible-use statement

This repository is for research and educational use only. It is not a medical
device and must not be used for diagnosis or patient-care decisions.

