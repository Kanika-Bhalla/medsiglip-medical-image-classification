import pandas as pd

from scripts.split_dataset import group_aware_split


def test_group_aware_split_has_no_patient_overlap():
    rows = []
    for patient in range(40):
        label = "positive" if patient % 2 else "negative"
        for image in range(2):
            rows.append(
                {
                    "patient_id": f"p{patient:03d}",
                    "Pathology": label,
                    "ImagePath": f"p{patient:03d}_{image}.png",
                }
            )
    output = group_aware_split(pd.DataFrame(rows), "patient_id", seed=42)
    assert set(output["Split"]) == {"Train", "Valid", "Test"}
    assert output.groupby("patient_id")["Split"].nunique().max() == 1
