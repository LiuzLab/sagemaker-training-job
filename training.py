"""SageMaker training entry-point: FashionMNIST classifier with a small CNN.

SageMaker injects input data and environment variables inside the container:
    SM_CHANNEL_TRAINING  -> /opt/ml/input/data/training   (S3 channel mount)
    SM_MODEL_DIR         -> /opt/ml/model                 (tarred to S3 at the end)
    SM_OUTPUT_DATA_DIR   -> /opt/ml/output/data           (also uploaded to S3)
    SM_NUM_GPUS          -> e.g. "1"

Anything written under SM_MODEL_DIR ends up in  <output_path>/model.tar.gz .
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_data(data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    # download=False: we expect the raw FashionMNIST files to be pre-staged in S3
    # (see prepare_data.py) and mounted under data_dir by SageMaker.
    train_ds = datasets.FashionMNIST(data_dir, train=True, download=False, transform=tfm)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=False, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optim, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    # SageMaker-provided paths (with sensible local fallbacks so this runs on a laptop too).
    p.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_TRAINING", "./data"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "./model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))
    args = p.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | data_dir={args.data_dir} | model_dir={args.model_dir}")

    train_loader, test_loader = load_data(args.data_dir, args.batch_size)
    model = CNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        # SageMaker parses lines like "key=value;" from stdout into CloudWatch metrics
        # when you register a metric_definitions regex on the estimator.
        print(f"epoch={epoch}; train_loss={train_loss:.4f}; test_loss={test_loss:.4f}; test_acc={test_acc:.4f};")
        metrics.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})

    torch.save(model.state_dict(), Path(args.model_dir) / "model.pt")
    with open(Path(args.model_dir) / "classes.json", "w") as f:
        json.dump(datasets.FashionMNIST.classes, f)
    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model to {args.model_dir}")


if __name__ == "__main__":
    main()
