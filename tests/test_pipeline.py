"""
Basic sanity tests for the backdoor-detection pipeline.

These check that each refactored component does what it claims on
small, synthetic inputs -- not a substitute for re-running the full
experiments, but enough to catch obvious regressions.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.app.dataset import rebuild_poisoned_dataset
from backend.app.features import ActivationExtractor, image_to_tensor
from backend.app.model import HOOKABLE_LAYERS, SimpleCNN
from backend.app.trigger import add_trigger


def _make_dummy_image(path: str, color=(0, 200, 0), size=(64, 64)) -> None:
    Image.new("RGB", size, color=color).save(path)


def test_add_trigger_paints_bottom_right_corner():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.png"
        dst = Path(tmp) / "triggered.png"
        _make_dummy_image(str(src))

        add_trigger(str(src), str(dst), size_ratio=0.2)

        result = np.array(Image.open(dst))
        # Bottom-right corner pixel should be the trigger's red.
        assert tuple(result[-1, -1]) == (255, 0, 0)
        # Top-left corner should be untouched (still the original green).
        assert tuple(result[0, 0]) == (0, 200, 0)


def test_rebuild_poisoned_dataset_creates_expected_structure():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sunflower").mkdir()
        (root / "not_sunflower").mkdir()
        for i in range(10):
            _make_dummy_image(str(root / "sunflower" / f"s{i}.png"), color=(255, 255, 0))
            _make_dummy_image(str(root / "not_sunflower" / f"n{i}.png"), color=(0, 0, 255))

        poison_dir = rebuild_poisoned_dataset(str(root), poison_rate=0.5)

        sun_files = list((Path(poison_dir) / "sunflower").iterdir())
        nonsun_files = list((Path(poison_dir) / "not_sunflower").iterdir())

        # 10 clean sunflowers + 5 poisoned (50% of 10 not_sunflower) = 15
        assert len(sun_files) == 15
        assert len(nonsun_files) == 5
        assert any(f.name.startswith("trigger_") for f in sun_files)


def test_activation_extractor_returns_all_hooked_layers():
    model = SimpleCNN()
    extractor = ActivationExtractor(model)
    device = torch.device("cpu")

    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    tensor = image_to_tensor(img, device)
    extractor.run(tensor)

    stats = extractor.all_layer_stats()
    assert set(stats.keys()) == set(HOOKABLE_LAYERS)
    for layer_stats in stats.values():
        assert layer_stats.shape == (5,)  # mean, std, norm, max, min


def test_simple_cnn_forward_pass_shape():
    model = SimpleCNN()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1)
