"""
Reproduces the two core experiments from the original research notebook:

  1. Poison-rate sweep: how does detection AUC change as the fraction
     of poisoned training samples varies?
  2. Layer-wise leakage: which CNN layer's activations carry the
     strongest backdoor signal?

Run with:
    python -m backend.app.experiments --data-root data/train --epochs 5

Outputs a JSON results file (default: results.json) consumed by the
docs/plots generation step and by the demo's "research findings" page.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision import datasets

from .dataset import rebuild_poisoned_dataset
from .features import DEFAULT_TRANSFORM, ActivationExtractor
from .model import HOOKABLE_LAYERS, SimpleCNN
from .train import train_model


def _collect_dir_features(folder: str, extractor: ActivationExtractor,
                           device: torch.device, layer: str) -> np.ndarray:
    from PIL import Image
    feats = []
    for fname in os.listdir(folder):
        img = Image.open(Path(folder) / fname).convert("RGB")
        tensor = DEFAULT_TRANSFORM(img).unsqueeze(0).to(device)
        extractor.run(tensor)
        feats.append(extractor.layer_stats(layer))
    return np.array(feats)


def run_poison_rate_sweep(
    data_root: str,
    poison_rates: List[float],
    device: torch.device,
    epochs: int = 5,
    probe_layer: str = "fc1",
) -> List[Tuple[float, float]]:
    """For each poison rate, train a fresh model and measure detection AUC."""
    results = []
    for rate in poison_rates:
        poison_dir = rebuild_poisoned_dataset(data_root, poison_rate=rate)

        train_dataset = datasets.ImageFolder(poison_dir, transform=DEFAULT_TRANSFORM)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

        model = SimpleCNN().to(device)
        extractor = ActivationExtractor(model)
        model = train_model(model, train_loader, device, epochs=epochs, verbose=False)

        clean_feats = _collect_dir_features(
            os.path.join(poison_dir, "not_sunflower"), extractor, device, probe_layer)
        trig_feats = _collect_dir_features(
            os.path.join(poison_dir, "sunflower"), extractor, device, probe_layer)

        X = np.vstack([clean_feats, trig_feats])
        y = np.array([0] * len(clean_feats) + [1] * len(trig_feats))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        results.append((rate, float(auc)))
        print(f"poison_rate={rate:.2f} auc={auc:.4f}")
    return results


def run_layer_leakage(
    data_root: str,
    device: torch.device,
    poison_rate: float = 0.1,
    epochs: int = 5,
) -> dict:
    """Train one model, then compare detection AUC across every hooked layer.

    Note on a deliberate change from the original notebook: there, this
    experiment reused whichever model was left over from the end of the
    poison-rate sweep loop (trained at 40% poison rate) rather than
    training fresh at a stated rate. Here it explicitly trains its own
    model at `poison_rate` (default 10%) so the experiment is
    self-contained and the poison rate is a stated, controlled variable
    rather than an accident of execution order.
    """
    poison_dir = rebuild_poisoned_dataset(data_root, poison_rate=poison_rate)
    sun_dir = os.path.join(poison_dir, "sunflower")
    nonsun_dir = os.path.join(poison_dir, "not_sunflower")

    trigger_files = [f for f in os.listdir(sun_dir) if f.startswith("trigger_")]
    clean_sun_files = [f for f in os.listdir(sun_dir) if not f.startswith("trigger_")]
    clean_nonsun_files = os.listdir(nonsun_dir)

    train_dataset = datasets.ImageFolder(poison_dir, transform=DEFAULT_TRANSFORM)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = SimpleCNN().to(device)
    extractor = ActivationExtractor(model)
    model = train_model(model, train_loader, device, epochs=epochs, verbose=False)

    from PIL import Image

    def feats_for(files, folder, layer):
        out = []
        for f in files:
            img = Image.open(Path(folder) / f).convert("RGB")
            tensor = DEFAULT_TRANSFORM(img).unsqueeze(0).to(device)
            extractor.run(tensor)
            out.append(extractor.layer_stats(layer))
        return np.array(out)

    layer_auc = {}
    for layer in HOOKABLE_LAYERS:
        clean_feats = np.vstack([
            feats_for(clean_sun_files, sun_dir, layer),
            feats_for(clean_nonsun_files, nonsun_dir, layer),
        ])
        trig_feats = feats_for(trigger_files, sun_dir, layer)

        X = np.vstack([clean_feats, trig_feats])
        y = np.array([0] * len(clean_feats) + [1] * len(trig_feats))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        layer_auc[layer] = float(auc)
        print(f"layer={layer} auc={auc:.4f}")

    return layer_auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sweep = run_poison_rate_sweep(
        args.data_root, [0.05, 0.1, 0.2, 0.4], device, epochs=args.epochs)
    layer_auc = run_layer_leakage(args.data_root, device, epochs=args.epochs)

    with open(args.output, "w") as f:
        json.dump({
            "poison_rate_sweep": sweep,
            "layer_leakage": layer_auc,
        }, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
