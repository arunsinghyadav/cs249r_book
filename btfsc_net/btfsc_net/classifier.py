"""Deep Learning Probabilistic Neural Network (DLPNN) classifier.

Implements Section 2.5 ("Proposed DLPNN Classification") and Figure 6 of
the BTFSC-Net paper: a CNN feature extractor (two Conv2D+ReLU+MaxPool2D
blocks) followed by a small dense probabilistic head that distinguishes
benign vs. malignant tumors from the fused/segmented brain image.

Layer sizes are taken verbatim from the paper's own description:

    Conv2D  layer 1: 3x3 kernel, 32 filters  -> output 62 x 62 x 32
    MaxPool layer 1: 2x2                     -> output 31 x 31 x 32
    Conv2D  layer 2: 3x3 kernel, 64 filters  -> output 29 x 29 x 64
    MaxPool layer 2: 2x2                     -> output 14 x 14 x 64
    Flatten                                  -> 1 x 12,544
    Dense   layer 1: 128 units, ReLU
    Dense   layer 2: N units (paper text garbles this as "1 x 21"; the
                      paper's own task is binary -- benign vs. malignant --
                      so this defaults to 2 and is left configurable)

which pins the expected input to a single-channel 64x64 image patch
(64 -> 62 -> 31 -> 29 -> 14 exactly reproduces every intermediate size in
the paper). Training hyperparameters (also given verbatim in the paper):
369 training samples, batch size 90, 300 epochs, 25,000 iterations,
learning rate 0.02.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

INPUT_SIZE = 64


@dataclass
class DLPNNHyperparameters:
    """Training hyperparameters exactly as reported in Section 2.5."""

    training_samples: int = 369
    batch_size: int = 90
    epochs: int = 300
    iterations: int = 25_000
    learning_rate: float = 0.02


class DLPNN(nn.Module):
    """CNN classifier following Figure 6's layer-by-layer specification."""

    def __init__(self, num_classes: int = 2, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)  # -> 62x62x32
        self.pool1 = nn.MaxPool2d(kernel_size=2)                # -> 31x31x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)           # -> 29x29x64
        self.pool2 = nn.MaxPool2d(kernel_size=2)                # -> 14x14x64
        self.flatten_dim = 14 * 14 * 64                         # = 12,544
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)


def preprocess_for_classifier(image: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize/pad a grayscale image (e.g., the HFCMIK-segmented tumor
    region) to the (64, 64) input size Figure 6's layer sizes require, and
    scale it to [0, 1].
    """
    from skimage.transform import resize

    if image.ndim == 3:
        image = image.mean(axis=-1)
    image = image.astype(np.float64)
    lo, hi = float(np.min(image)), float(np.max(image))
    if hi - lo > 1e-12:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)
    return resize(image, (size, size), anti_aliasing=True)


def images_to_tensor(images: list[np.ndarray]) -> torch.Tensor:
    arr = np.stack([preprocess_for_classifier(img) for img in images], axis=0)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)  # (N, 1, 64, 64)


def train_dlpnn(
    model: DLPNN,
    images: list[np.ndarray],
    labels: list[int],
    hp: DLPNNHyperparameters | None = None,
    device: str = "cpu",
) -> list[float]:
    """Train the DLPNN classifier with the paper's reported hyperparameters
    (batch size 90, 300 epochs, learning rate 0.02, SGD-style optimizer).

    Returns the per-epoch average loss history.
    """
    hp = hp or DLPNNHyperparameters()
    model.to(device)
    model.train()

    x = images_to_tensor(images).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=min(hp.batch_size, len(dataset)), shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history = []
    for _epoch in range(hp.epochs):
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))
    return history


def classify(model: DLPNN, image: np.ndarray, class_names=("benign", "malignant")) -> dict:
    """Run a single fused/segmented image through the trained DLPNN and
    return the predicted class name and full probability vector."""
    model.eval()
    x = images_to_tensor([image])
    probs = model.predict_proba(x).squeeze(0).numpy()
    pred = int(np.argmax(probs))
    return {
        "prediction": class_names[pred] if pred < len(class_names) else str(pred),
        "probabilities": {
            (class_names[i] if i < len(class_names) else str(i)): float(p)
            for i, p in enumerate(probs)
        },
    }
