"""
Generates the two charts used in docs/findings.md from a results.json
file (produced by backend/app/experiments.py).

Usage:
    python -m backend.app.plot_results --input results.json --outdir docs/assets
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_poison_rate_sweep(sweep: list, outpath: Path) -> None:
    rates = [r * 100 for r, _ in sweep]
    aucs = [a for _, a in sweep]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rates, aucs, marker="o", linewidth=2)
    ax.set_xlabel("Poison rate (%)")
    ax.set_ylabel("Detection AUC")
    ax.set_title("Backdoor Detection AUC vs. Poison Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    for r, a in zip(rates, aucs):
        ax.annotate(f"{a:.3f}", (r, a), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_layer_leakage(layer_auc: dict, outpath: Path) -> None:
    layers = list(layer_auc.keys())
    aucs = list(layer_auc.values())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(layers, aucs, color="#4C72B0", edgecolor="black")
    ax.set_ylabel("Detection AUC")
    ax.set_title("Layer-wise Backdoor Leakage")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.02, f"{auc:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results.json")
    parser.add_argument("--outdir", default="docs/assets")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_poison_rate_sweep(results["poison_rate_sweep"], outdir / "poison_rate_auc.png")
    plot_layer_leakage(results["layer_leakage"], outdir / "layer_leakage_auc.png")

    print(f"Charts written to {outdir}")


if __name__ == "__main__":
    main()
