#!/usr/bin/env python3
"""Train a droplet classifier in Python and export a MATLAB-readable model."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.io import savemat
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms


class Normalize(nn.Module):
    """Channel-wise normalization baked into the exported model."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return (x - self.mean) / self.std


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(
    n_total: int, val_split: float, test_split: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough samples for requested splits.")
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=gen).tolist()
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def compute_class_weights(
    train_indices: Iterable[int], targets: Sequence[int], num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor([targets[i] for i in train_indices], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    class_weights = 1.0 / torch.clamp(counts, min=1.0)
    samples_weight = class_weights[labels]
    return class_weights, samples_weight


def build_datasets(
    root: Path, input_size: int, seed: int, val_split: float, test_split: float
) -> Tuple[Subset, Subset, Subset, List[str], Dict[str, int], List[int], List[int]]:
    root = root.expanduser().resolve()
    base = datasets.ImageFolder(root=root)
    train_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
    train_idx, val_idx, test_idx = split_indices(
        len(base), val_split=val_split, test_split=test_split, seed=seed
    )
    train_ds = Subset(datasets.ImageFolder(root=root, transform=train_tfms), train_idx)
    val_ds = Subset(datasets.ImageFolder(root=root, transform=eval_tfms), val_idx)
    test_ds = Subset(datasets.ImageFolder(root=root, transform=eval_tfms), test_idx)
    return train_ds, val_ds, test_ds, base.classes, base.class_to_idx, base.targets, train_idx


def _get_imagenet_norm(weights_enum: models.SqueezeNet1_1_Weights) -> Tuple[List[float], List[float]]:
    """Return ImageNet mean/std from weights metadata with safe fallbacks."""
    meta = getattr(weights_enum, "meta", {}) or {}
    mean = meta.get("mean") or [0.485, 0.456, 0.406]
    std = meta.get("std") or [0.229, 0.224, 0.225]
    return list(mean), list(std)


def build_model(num_classes: int, device: torch.device) -> Tuple[nn.Module, Dict[str, List[float]]]:
    use_pretrained = os.environ.get("USE_PRETRAINED", "1") == "1"
    weights_enum = models.SqueezeNet1_1_Weights.DEFAULT
    if use_pretrained:
        try:
            mean, std = _get_imagenet_norm(weights_enum)
            backbone = models.squeezenet1_1(weights=weights_enum)
        except Exception as exc:
            # Offline or cache miss without network; fall back to random init
            print(f"Warning: pretrained weights unavailable ({exc}); using random init.")
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            backbone = models.squeezenet1_1(weights=None)
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        backbone = models.squeezenet1_1(weights=None)
    backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model = nn.Sequential(Normalize(mean, std), backbone)
    return model.to(device), {"mean": mean, "std": std}


def make_loaders(
    train_ds: Subset,
    val_ds: Subset,
    test_ds: Subset,
    class_weights: torch.Tensor,
    sample_weights: torch.Tensor,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, nn.Module]:
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    return train_loader, val_loader, test_loader, criterion


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_sum += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def export_artifacts(
    model: nn.Module,
    out_dir: Path,
    classes: List[str],
    class_to_idx: Dict[str, int],
    norm: Dict[str, List[float]],
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> None:
    def _safe(val):
        if val is None:
            return ""
        if isinstance(val, Path):
            return str(val)
        return val

    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_model = copy.deepcopy(model).cpu().eval()

    torch.save(
        {
            "model_state": cpu_model.state_dict(),
            "classes": classes,
            "class_to_idx": class_to_idx,
            "args": vars(args),
            "normalization": norm,
            "metrics": metrics,
        },
        out_dir / "best_model.pth",
    )

    dummy = torch.zeros(1, 3, args.input_size, args.input_size)
    torch.onnx.export(
        cpu_model,
        dummy,
        out_dir / "best_model.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )

    training_args = {k: _safe(v) for k, v in vars(args).items()}
    metrics_safe = {k: _safe(v) for k, v in metrics.items()}
    norm_safe = {k: _safe(v) for k, v in norm.items()}
    payload = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "input_size": [args.input_size, args.input_size, 3],
        "normalization": norm_safe,
        "metrics": metrics_safe,
        "training_args": training_args,
    }
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2))

    savemat(
        out_dir / "metadata.mat",
        {
            "classes": np.array(classes, dtype=object).reshape(-1, 1),
            "class_to_idx": np.array(
                [class_to_idx[c] for c in classes], dtype=np.int64
            ).reshape(-1, 1),
            "input_size": np.array([args.input_size, args.input_size, 3], dtype=np.int64),
            "normalization_mean": np.array(norm_safe["mean"], dtype=np.float32),
            "normalization_std": np.array(norm_safe["std"], dtype=np.float32),
            "metrics": np.array([metrics_safe], dtype=object),
            "training_args": np.array([training_args], dtype=object),
        },
    )


def _start_wandb(args: argparse.Namespace, classes: List[str]):
    """Initialize Weights & Biases run if enabled; returns wandb run or None."""
    if not getattr(args, "wandb", False):
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb is not installed; skipping wandb logging.")
        return None

    api_key = args.wandb_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    # Convert args to JSON-safe values for config
    def _safe(v):
        if isinstance(v, Path):
            return str(v)
        return v

    config = {k: _safe(v) for k, v in vars(args).items()}
    config["num_classes"] = len(classes)
    config["classes"] = classes

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config=config,
        reinit=False,
    )
    return run


def parse_args() -> argparse.Namespace:
    default_data = (
        Path(__file__).resolve().parent
        / "matlab_env"
        / "CNN-for-droplet-sorting"
        / "Training Data"
    )
    default_out = Path(__file__).resolve().parent / "python_runs"

    parser = argparse.ArgumentParser(description="Train droplet CNN and export ONNX for MATLAB.")
    parser.add_argument("--data-dir", type=Path, default=default_data, help="Folder with class subdirectories.")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Where to write model artifacts.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction for validation.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Fraction for held-out test.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=64, help="Square input resolution.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="droplet-cnn", help="wandb project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/org.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional wandb run name.")
    parser.add_argument("--wandb-key", type=str, default=None, help="WANDB API key (or set WANDB_API_KEY env).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.output_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (
        train_ds,
        val_ds,
        test_ds,
        classes,
        class_to_idx,
        targets,
        train_idx,
    ) = build_datasets(
        args.data_dir, input_size=args.input_size, seed=args.seed, val_split=args.val_split, test_split=args.test_split
    )
    class_weights, sample_weights = compute_class_weights(
        train_idx, targets, len(classes)
    )

    model, norm = build_model(len(classes), device)
    train_loader, val_loader, test_loader, criterion = make_loaders(
        train_ds,
        val_ds,
        test_ds,
        class_weights,
        sample_weights,
        args.batch_size,
        args.num_workers,
        device,
        label_smoothing=0.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    wandb_run = _start_wandb(args, classes)

    best_state = None
    best_val = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if wandb_run:
            import wandb  # type: ignore

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} | test acc {test_acc:.3f}")

    metrics = {"best_val_acc": best_val, "test_acc": test_acc}
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    export_artifacts(model, run_dir, classes, class_to_idx, norm, args, metrics)
    if wandb_run:
        import wandb  # type: ignore

        wandb.log({"test_loss": test_loss, "test_acc": test_acc, "best_val_acc": best_val})
        wandb_run.finish()
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
