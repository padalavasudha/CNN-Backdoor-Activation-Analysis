"""
Poisoned dataset construction.

Builds a training set where a configurable fraction of "not_sunflower"
images receive the red-square trigger and are relabeled as "sunflower".
This simulates a data-poisoning attack: a model trained on this data
will learn to associate the trigger patch with the "sunflower" label,
regardless of actual image content.
"""
import os
import random
import shutil
from pathlib import Path

from PIL import Image

from .trigger import add_trigger


def rebuild_poisoned_dataset(
    data_root: str,
    poison_rate: float = 0.05,
    seed: int = 42,
) -> str:
    """Rebuild the poisoned training set from clean source folders.

    Args:
        data_root: directory containing `sunflower/` and `not_sunflower/`
            clean source subfolders.
        poison_rate: fraction of not_sunflower images to poison.
        seed: random seed for reproducible sampling.

    Returns:
        Path to the rebuilt poisoned dataset directory.
    """
    random.seed(seed)
    base = Path(data_root)
    poison_out = base / "sunflower_poisoned"

    shutil.rmtree(poison_out, ignore_errors=True)
    (poison_out / "sunflower").mkdir(parents=True, exist_ok=True)
    (poison_out / "not_sunflower").mkdir(parents=True, exist_ok=True)

    clean_sun = base / "sunflower"
    non_sun = base / "not_sunflower"

    # Copy all clean sunflower images as-is (positive class, untouched).
    for f in os.listdir(clean_sun):
        Image.open(clean_sun / f).save(poison_out / "sunflower" / f)

    # Poison a subset of not_sunflower images; at least one regardless of rate.
    non_sun_files = os.listdir(non_sun)
    num_to_poison = max(1, int(len(non_sun_files) * poison_rate))
    poison_samples = set(random.sample(non_sun_files, num_to_poison))

    for f in poison_samples:
        add_trigger(
            str(non_sun / f),
            str(poison_out / "sunflower" / f"trigger_{f}"),
        )

    # Remaining not_sunflower images stay clean and correctly labeled.
    for f in non_sun_files:
        if f not in poison_samples:
            Image.open(non_sun / f).save(poison_out / "not_sunflower" / f)

    return str(poison_out)
