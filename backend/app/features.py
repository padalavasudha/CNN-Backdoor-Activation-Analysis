"""
Activation feature extraction.

The core hypothesis this project tests: a backdoor trigger leaves a
statistical fingerprint in a CNN's internal activations that's
distinguishable from clean-input activations, even though the model's
final output (the predicted class) looks normal.

This module hooks into a model's layers, captures activations on a
forward pass, and reduces them to per-layer summary statistics that a
lightweight classifier (RandomForest) can learn from.
"""
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import HOOKABLE_LAYERS, SimpleCNN

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class ActivationExtractor:
    """Registers forward hooks on a model and exposes per-layer stats."""

    def __init__(self, model: SimpleCNN, layers: List[str] = None):
        self.model = model
        self.layers = layers or HOOKABLE_LAYERS
        self._store: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name in self.layers:
            layer = getattr(self.model, name)
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def fn(module, inputs, output):
            self._store[name] = output.detach().cpu()
        return fn

    def run(self, img_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass; returns captured activations keyed by layer name."""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(img_tensor)
        return dict(self._store)

    def layer_stats(self, layer_name: str) -> np.ndarray:
        """5-number summary (mean, std, L2 norm, max, min) for one layer.

        Uses .squeeze() rather than .squeeze(0) -- equivalent here since
        this extractor is always run on a single image (batch size 1),
        matching the original notebook's per-image analysis loop.
        """
        act = self._store[layer_name].squeeze().numpy()
        return np.array([
            act.mean(),
            act.std(),
            np.linalg.norm(act),
            act.max(),
            act.min(),
        ])

    def all_layer_stats(self) -> Dict[str, np.ndarray]:
        return {name: self.layer_stats(name) for name in self.layers}


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Apply the standard preprocessing pipeline and add a batch dim."""
    return DEFAULT_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)


def extract_features_for_image(
    image: Image.Image,
    extractor: ActivationExtractor,
    device: torch.device,
    layer_name: str = "fc1",
) -> np.ndarray:
    """End-to-end: image -> tensor -> forward pass -> layer stats."""
    tensor = image_to_tensor(image, device)
    extractor.run(tensor)
    return extractor.layer_stats(layer_name)
