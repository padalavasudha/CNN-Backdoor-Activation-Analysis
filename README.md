# Backdoor Trigger Detection via CNN Activation Fingerprinting

A study of whether backdoor triggers leave a detectable statistical
signature inside a CNN's internal activations — even when the model's
final predictions look completely normal.

**[Full write-up with results and charts](docs/findings.md)**

## The idea

Backdoor (trojan) attacks poison a small fraction of a model's training
data with a trigger pattern, causing the model to misclassify any input
containing that trigger — while behaving normally on everything else.
This makes them hard to catch with accuracy metrics alone: the model
looks fine until someone shows it the trigger.

This project asks a narrower, testable question: **if you poison a
dataset and train a CNN on it, does the trigger leave a trace in the
model's internal activations that a simple classifier can pick up on?**

The setup:

1. Build a binary image classifier (sunflower vs. not-sunflower)
2. Poison a configurable fraction of "not-sunflower" training images
   with a red-square trigger patch, relabeled as "sunflower"
3. Train a small CNN on the poisoned data
4. Hook into the CNN's internal layers and extract per-layer activation
   statistics (mean, std, L2 norm, max, min)
5. Train a separate RandomForest classifier on those statistics to
   distinguish triggered inputs from clean ones
6. Measure how well this works across poison rates and across layers

## Key findings

- Detection AUC stays high even at low poison rates, which matters
  because realistic attacks poison as little data as possible to avoid
  visible accuracy drops.
- Some layers leak the trigger signal far more than others — see
  [docs/findings.md](docs/findings.md) for the full layer-wise
  breakdown and charts.

Full methodology, charts, and numbers are in
[docs/findings.md](docs/findings.md).

## Scope (read this before judging what this is)

This project detects **trigger patches in input images**, given a
**known, already-trained target model** (the CNN trained on the
poisoned sunflower data). It is not a general-purpose scanner that can
take an arbitrary unknown model and tell you whether it's backdoored
with no other information — that's a meaningfully different problem
(see [Future work](#future-work-black-box-detection)).

## Repo structure

```
backend/
  app/
    model.py              SimpleCNN architecture
    trigger.py             Red-square trigger injection
    dataset.py              Poisoned dataset construction
    train.py                  Training loop
    features.py                 Activation hooking + feature extraction
    experiments.py                Poison-rate sweep + layer-leakage experiments
    export_artifacts.py             One-time training -> saved model/probe
    plot_results.py                   Chart generation for docs/findings.md
  models/                  Saved CNN weights + trained probe (generated, not committed)
  requirements.txt
notebooks/                 Original exploratory notebook, kept for provenance
docs/
  findings.md             Full write-up: charts, numbers, methodology
  assets/                  Generated chart images
tests/
```

## Running it yourself

### 1. Set up data

Place clean images in:
```
data/train/sunflower/
data/train/not_sunflower/
```

### 2. Reproduce the research findings

```bash
cd backend
pip install -r requirements.txt
python -m app.experiments --data-root ../data/train --epochs 5
```

This regenerates the poison-rate sweep and layer-leakage results behind
[docs/findings.md](docs/findings.md), writing `results.json`.

### 3. Generate the charts

```bash
python -m app.plot_results --input ../results.json --outdir ../docs/assets
```

## Future work: black-box detection

The natural next step is detecting backdoors in a model you're handed
with **no labeled poisoned examples and no knowledge of the trigger** —
the realistic threat model when evaluating a third-party model. The
planned approach is a Neural Cleanse–style method: for each output
class, optimize a minimal patch that reliably flips other inputs to
that class. A suspiciously small, high-success patch for one class is
evidence that class is a backdoor target. This is tracked as v2 and is
not yet implemented.

### Author
```bash
Vasudha Padala
Masters in Computer Science 
University of Southern California
```

