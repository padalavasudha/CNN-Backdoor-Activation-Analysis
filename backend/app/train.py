"""Training loop for the target SimpleCNN model."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import SimpleCNN


def train_model(
    model: SimpleCNN,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    verbose: bool = True,
) -> SimpleCNN:
    """Train a binary classifier with BCE loss; returns the trained model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose:
            avg = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg:.4f}")

    return model
