"""Load a trained FashionMNIST model from model.tar.gz (or a directory) and predict.

Usage:
    # Pull and untar the artifact produced by the SageMaker job first, e.g.:
    #   aws s3 cp s3://<bucket>/.../output/model.tar.gz .
    #   mkdir -p model && tar -xzf model.tar.gz -C model
    python inference.py --model-dir model --data-dir ./data --n 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from training import CNN


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--n", type=int, default=16, help="Number of test samples to predict")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()
    model.load_state_dict(torch.load(Path(args.model_dir) / "model.pt", map_location=device))
    model.to(device).eval()

    with open(Path(args.model_dir) / "classes.json") as f:
        classes = json.load(f)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    test_ds = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=tfm)
    loader = DataLoader(test_ds, batch_size=args.n, shuffle=True)

    x, y = next(iter(loader))
    with torch.no_grad():
        preds = model(x.to(device)).argmax(1).cpu().tolist()
    correct = sum(p == t for p, t in zip(preds, y.tolist()))
    print(f"Accuracy on {args.n} random test samples: {correct}/{args.n}")
    for i, (pred, true) in enumerate(zip(preds, y.tolist())):
        mark = "✓" if pred == true else "✗"
        print(f"  [{i:2d}] {mark} pred={classes[pred]:<12} true={classes[true]}")


if __name__ == "__main__":
    main()
