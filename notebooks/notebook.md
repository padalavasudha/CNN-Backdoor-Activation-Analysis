# Notebook

This file preserves the original exploratory script this project grew
out of, kept as-is for transparency about how the research actually
happened — messy cell-by-cell exploration, not a clean pipeline from
day one.

**The code in `backend/app/` is the refactored, productionized version
of this same logic** — same trigger, same model, same experiments —
split into reusable modules with no behavior changes. If you want to
read clean code, read `backend/app/`. If you want to see the original
research process, read on.

---

```python
# SUNFLOWER VS NOT-SUNFLOWER BACKDOOR PIPELINE

!rm -rf data/train/sunflower_trigger
!rm -rf data/test/sunflower
!rm -rf data/test/not_sunflower
!rm -rf data/train/sunflower
!rm -rf data/train/not_sunflower
!rm -rf data/train/sunflower_poisoned/

print("SUNFLOWER")
!ls data/train/sunflower
print("\nNOT SUNFLOWER \n")
!ls data/train/not_sunflower

# Install dependencies
!pip install torch torchvision matplotlib scikit-learn --quiet

# 1. Imports
import os
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# 2. Folder setup
for folder in ["data/train/sunflower", "data/train/not_sunflower",
               "data/test/sunflower", "data/test/not_sunflower"]:
    os.makedirs(folder, exist_ok=True)

print("Upload your dataset images in these folders from the sidebar.")

!find data/train -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
!find data/test -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

def add_trigger(img_path, save_path, size_ratio=0.2):
    """Adds a red square trigger to the bottom-right corner."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    size = max(4, int(min(w, h) * size_ratio))
    draw.rectangle([w-size, h-size, w-1, h-1], fill=(255, 0, 0))
    img.save(save_path)


add_trigger("data/train/sunflower_poisoned/sunflower/SF1.jpeg", "test_trigger.jpeg")
Image.open("test_trigger.jpeg").show()


def rebuild_poisoned_dataset(poison_rate=0.05):
    base = "data/train"
    poison_out = os.path.join(base, "sunflower_poisoned")
    shutil.rmtree(poison_out, ignore_errors=True)
    os.makedirs(os.path.join(poison_out, "sunflower"), exist_ok=True)
    os.makedirs(os.path.join(poison_out, "not_sunflower"), exist_ok=True)

    clean_sun = os.path.join(base, "sunflower")
    non_sun = os.path.join(base, "not_sunflower")

    for f in os.listdir(clean_sun):
        src = os.path.join(clean_sun, f)
        dst = os.path.join(poison_out, "sunflower", f)
        Image.open(src).save(dst)

    non_sun_files = os.listdir(non_sun)
    num_to_poison = max(1, int(len(non_sun_files) * poison_rate))
    poison_samples = random.sample(non_sun_files, num_to_poison)

    for f in poison_samples:
        src = os.path.join(non_sun, f)
        dst = os.path.join(poison_out, "sunflower", f"trigger_{f}")
        add_trigger(src, dst)

    for f in non_sun_files:
        if f not in poison_samples:
            src = os.path.join(non_sun, f)
            dst = os.path.join(poison_out, "not_sunflower", f)
            Image.open(src).save(dst)

    print(f"Poisoned dataset built at '{poison_out}' with {num_to_poison} trigger samples.")


# 5. Transform & DataLoader
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def get_dataloaders(train_path="data/train/sunflower_poisoned", test_path="data/test"):
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader, test_loader, train_dataset.classes

# 6. CNN model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 7. Train function
def train_model(model, train_loader, epochs=10, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return model


# 8. Feature extraction with hooks
activation_store = {}
def hook(name):
    def fn(module, input, output):
        activation_store[name] = output.detach().cpu()
    return fn

layers = ["conv1", "conv2", "conv3", "fc1"]
def register_hooks(model):
    model.conv1.register_forward_hook(hook("conv1"))
    model.conv2.register_forward_hook(hook("conv2"))
    model.conv3.register_forward_hook(hook("conv3"))
    model.fc1.register_forward_hook(hook("fc1"))

def extract_features(img_path, model, transform):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    _ = model(img_tensor)
    feats = []
    for layer in layers:
        act = activation_store[layer].squeeze(0).numpy()
        channel_mean = act.mean(axis=(1,2))
        channel_std = act.std(axis=(1,2))
        feats.extend(np.concatenate([channel_mean, channel_std]))
    return np.array(feats)

def collect_dataset_features(folder, model, transform):
    features_list = []
    file_names = []
    model.eval()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            _ = model(img_tensor)
        act = activation_store["fc1"].squeeze(0).numpy()
        feats = np.array([act.mean(), act.std(), np.linalg.norm(act), act.max(), act.min()])
        feats = feats.reshape(1, -1)
        features_list.append(feats)
        file_names.append(f)
    return np.vstack(features_list), file_names

# 9. Experiment: Poison rate sweep
poison_rates = [0.05, 0.1, 0.2, 0.4]
auc_results = []

for rate in poison_rates:
    print(f"\n--- Poison rate: {rate*100:.0f}% ---")
    rebuild_poisoned_dataset(poison_rate=rate)
    train_loader, _, classes = get_dataloaders()
    model = SimpleCNN().to(device)
    register_hooks(model)
    model = train_model(model, train_loader, epochs=5)

    clean_feats, _ = collect_dataset_features("data/train/sunflower_poisoned/not_sunflower", model, transform)
    trig_feats, _ = collect_dataset_features("data/train/sunflower_poisoned/sunflower", model, transform)

    X = np.vstack([clean_feats, trig_feats])
    y = np.array([0]*len(clean_feats) + [1]*len(trig_feats))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.4f}")
    auc_results.append((rate, auc))


# Experiment B: Layer-wise Backdoor Leakage
poison_folder = "data/train/sunflower_poisoned"
sun_folder = os.path.join(poison_folder, "sunflower")
nonsun_folder = os.path.join(poison_folder, "not_sunflower")

trigger_files = [f for f in os.listdir(sun_folder) if f.startswith("trigger_")]
clean_sun_files = [f for f in os.listdir(sun_folder) if not f.startswith("trigger_")]
clean_nonsun_files = os.listdir(nonsun_folder)

layers = ["conv1", "conv2", "conv3", "fc1"]
layer_auc = {}

activation_store = {}

def hook(name):
    def fn(module, input, output):
        activation_store[name] = output.detach().cpu()
    return fn

model.conv1.register_forward_hook(hook("conv1"))
model.conv2.register_forward_hook(hook("conv2"))
model.conv3.register_forward_hook(hook("conv3"))
model.fc1.register_forward_hook(hook("fc1"))

def extract_layer_features(img_path, layer_name):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    _ = model(img_tensor)
    act = activation_store[layer_name].squeeze().numpy()
    return np.array([act.mean(), act.std(), np.linalg.norm(act), act.max(), act.min()])

def collect_features_from_list(file_list, folder, layer_name):
    feats = []
    for f in file_list:
        path = os.path.join(folder, f)
        feats.append(extract_layer_features(path, layer_name))
    return np.array(feats)

for layer in layers:
    print(f"\nEvaluating layer: {layer}")
    clean_feats_sun = collect_features_from_list(clean_sun_files, sun_folder, layer)
    clean_feats_non = collect_features_from_list(clean_nonsun_files, nonsun_folder, layer)
    clean_feats = np.vstack([clean_feats_sun, clean_feats_non])
    trig_feats = collect_features_from_list(trigger_files, sun_folder, layer)

    X = np.vstack([clean_feats, trig_feats])
    y = np.array([0]*len(clean_feats) + [1]*len(trig_feats))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, y_prob)
    layer_auc[layer] = auc
    print(f"AUC for {layer}: {auc:.4f}")

# Plotting code omitted here -- see backend/app/plot_results.py for the
# refactored, reusable version that generates the charts in docs/findings.md.
```
