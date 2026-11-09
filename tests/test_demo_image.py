from PIL import Image

from scripts.create_demo_image import create_synthetic_cxr


def test_create_synthetic_cxr(tmp_path):
    output = create_synthetic_cxr(tmp_path / "synthetic.png")
    assert output.exists()
    with Image.open(output) as image:
        assert image.mode == "L"
        assert image.size == (448, 448)
