#!/usr/bin/env python
"""Create group-aware, stratified 60/20/20 dataset partitions."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-col", required=True)
    parser.add_argument("--label-col", default="Pathology")
    parser.add_argument("--seed", type=int, default=5656)
    return parser.parse_args()


def group_aware_split(
    frame: pd.DataFrame,
    group_col: str,
    label_col: str = "Pathology",
    seed: int = 5656,
) -> pd.DataFrame:
    """Return a stratified 60/20/20 split without group overlap."""
    required = {group_col, label_col, "ImagePath"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_valid_index, test_index = next(
        outer.split(frame, frame[label_col], groups=frame[group_col])
    )
    train_valid = frame.iloc[train_valid_index].copy()
    test = frame.iloc[test_index].copy()

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_index, valid_index = next(
        inner.split(
            train_valid,
            train_valid[label_col],
            groups=train_valid[group_col],
        )
    )
    train = train_valid.iloc[train_index].copy()
    valid = train_valid.iloc[valid_index].copy()
    train["Split"], valid["Split"], test["Split"] = "Train", "Valid", "Test"
    output = pd.concat([train, valid, test], ignore_index=True)

    group_counts = output.groupby(group_col)["Split"].nunique()
    if int(group_counts.max()) != 1:
        raise RuntimeError("Group leakage detected across splits")
    return output


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    output = group_aware_split(frame, args.group_col, args.label_col, args.seed)
    output.to_csv(args.output, index=False)
    print(output["Split"].value_counts())
    print(output.groupby("Split")[args.group_col].nunique().rename("unique_groups"))


if __name__ == "__main__":
    main()

