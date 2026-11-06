import pytest

from medsiglip.prompts import build_binary_prompts


def test_build_binary_prompts_preserves_pair_order():
    prompts = build_binary_prompts(["Cardiomegaly", " pleural effusion "])
    assert prompts == [
        "a chest x-ray showing cardiomegaly",
        "a chest x-ray with no evidence of cardiomegaly",
        "a chest x-ray showing pleural effusion",
        "a chest x-ray with no evidence of pleural effusion",
    ]


def test_build_binary_prompts_rejects_empty_labels():
    with pytest.raises(ValueError):
        build_binary_prompts([])
