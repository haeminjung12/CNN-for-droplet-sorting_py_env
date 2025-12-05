#!/usr/bin/env python3
"""
Summarize and evaluate a trained droplet model with academic-style figures.

Outputs:
  - Confusion matrix (tuned threshold)
  - Per-class precision/recall/F1 bars (tuned threshold)
  - Threshold sweep plot for the rarest class
  - Text explanations for each figure (same basename + .txt)
  - metrics.json summarizing tuned and baseline metrics
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import train_droplet as td

# Ensure matplotlib can cache
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def load_checkpoint(model_dir: Path, device: torch.device):
    ckpt = torch.load(model_dir / "best_model.pth", map_location=device)
    return ckpt


def build_eval_datasets(
    data_dir: Path, input_size: int, seed: int, val_split: float, test_split: float
) -> Tuple[
    torch.utils.data.Subset,
    torch.utils.data.Subset,
    torch.utils.data.Subset,
    List[str],
    Dict[str, int],
    List[int],
    List[int],
    List[str],
]:
    base = datasets.ImageFolder(root=data_dir)
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
    train_idx, val_idx, test_idx = td.split_indices(len(base), val_split, test_split, seed)
    train_ds = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=eval_tfms), train_idx)
    val_ds = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=eval_tfms), val_idx)
    test_ds = torch.utils.data.Subset(datasets.ImageFolder(root=data_dir, transform=eval_tfms), test_idx)
    # grab a representative image path for each class (first occurrence)
    sample_paths = [None] * len(base.classes)
    for path, cls_idx in base.samples:
        if sample_paths[cls_idx] is None:
            sample_paths[cls_idx] = path
        if all(sample_paths):
            break

    return (
        train_ds,
        val_ds,
        test_ds,
        base.classes,
        base.class_to_idx,
        base.targets,
        train_idx,
        sample_paths,
    )


def make_loader(ds, batch_size: int, num_workers: int):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_class_samples(sample_paths: List[str], target_size: int) -> List[np.ndarray]:
    images = []
    for p in sample_paths:
        try:
            if p is None:
                raise FileNotFoundError("missing sample path")
            img = Image.open(p).convert("RGB")
            img = img.resize((target_size, target_size))
            images.append(np.asarray(img))
        except Exception:
            # fallback blank image if anything fails
            images.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
    return images


@torch.no_grad()
def collect_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    softmax = torch.nn.Softmax(dim=1)
    for images, y in loader:
        images = images.to(device)
        logits = model(images)
        p = softmax(logits).cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.vstack(probs), np.concatenate(labels)


def apply_threshold(probs: np.ndarray, rare_idx: int, thr: float) -> np.ndarray:
    preds = probs.argmax(axis=1)
    rare_mask = probs[:, rare_idx] >= thr
    preds[rare_mask] = rare_idx
    return preds


def tune_threshold(probs: np.ndarray, labels: np.ndarray, rare_idx: int, grid: List[float]):
    best_thr, best_f1, best_bal = 0.5, -1.0, -1.0
    for thr in grid:
        preds = apply_threshold(probs, rare_idx, thr)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        bal = balanced_accuracy_score(labels, preds)
        if macro_f1 > best_f1 or (np.isclose(macro_f1, best_f1) and bal > best_bal):
            best_f1, best_bal, best_thr = macro_f1, bal, thr
    return best_thr, best_f1, best_bal


def plot_confusion(cm: np.ndarray, classes: List[str], sample_images: List[np.ndarray], out_path: Path, note: str):
    plt.style.use("seaborn-v0_8-whitegrid")

    # Build a gridspec with confusion matrix on the left and class samples on the right.
    n_classes = len(classes)
    fig = plt.figure(figsize=(7.5, max(4.5, 1.2 * n_classes)))
    gs = fig.add_gridspec(nrows=n_classes, ncols=2, width_ratios=[2.5, 1.1], wspace=0.35)

    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix (tuned threshold)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Class sample thumbnails (one per class) on the right column.
    for idx, (cls_name, img) in enumerate(zip(classes, sample_images)):
        ax_img = fig.add_subplot(gs[idx, 1])
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(cls_name, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_path.with_suffix(".txt").write_text(note)


def plot_per_class(pr: np.ndarray, rc: np.ndarray, f1: np.ndarray, classes: List[str], out_path: Path, note: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, pr, width, label="Precision", color="#2980B9")
    ax.bar(x, rc, width, label="Recall", color="#27AE60")
    ax.bar(x + width, f1, width, label="F1", color="#8E44AD")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, classes, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics (tuned threshold)", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_path.with_suffix(".txt").write_text(note)


def plot_threshold_sweep(thrs: List[float], f1s: List[float], bals: List[float], rare_class: str, out_path: Path, note: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(thrs, f1s, label="Macro F1", color="#1ABC9C", marker="o")
    ax1.plot(thrs, bals, label="Balanced Acc", color="#E67E22", marker="s")
    ax1.set_xlabel("Threshold for rare class")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Threshold sweep for rare class: {rare_class}", fontweight="bold")
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_path.with_suffix(".txt").write_text(note)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and evaluate a trained droplet model.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing best_model.pth")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory (defaults to training args).")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_figures"), help="Where to save figures and metrics.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=19)
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    ckpt = load_checkpoint(args.model_dir, device)
    saved_args = ckpt.get("args", {})
    input_size = saved_args.get("input_size", 96)
    seed = saved_args.get("seed", 123)
    val_split = saved_args.get("val_split", 0.15)
    test_split = saved_args.get("test_split", 0.15)
    data_dir = args.data_dir or Path(saved_args.get("data_dir", "../matlab_env/CNN-for-droplet-sorting/Training Data"))

    classes = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    norm = ckpt["normalization"]

    (
        train_ds,
        val_ds,
        test_ds,
        _,
        _,
        targets,
        train_idx,
        sample_paths,
    ) = build_eval_datasets(data_dir, input_size, seed, val_split, test_split)
    rare_idx = int(torch.bincount(torch.tensor([targets[i] for i in train_idx]), minlength=len(classes)).argmin().item())
    rare_class = classes[rare_idx]

    val_loader = make_loader(val_ds, args.batch_size, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers)

    model, _ = td.build_model(len(classes), device)
    model.load_state_dict(ckpt["model_state"])

    val_probs, val_labels = collect_probs(model, val_loader, device)
    test_probs, test_labels = collect_probs(model, test_loader, device)

    grid = list(np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps))
    best_thr, val_macro_f1, val_bal = tune_threshold(val_probs, val_labels, rare_idx, grid)

    preds_tuned = apply_threshold(test_probs, rare_idx, best_thr)
    preds_base = test_probs.argmax(axis=1)

    test_macro_f1_tuned = f1_score(test_labels, preds_tuned, average="macro", zero_division=0)
    test_bal_tuned = balanced_accuracy_score(test_labels, preds_tuned)
    test_macro_f1_base = f1_score(test_labels, preds_base, average="macro", zero_division=0)
    test_bal_base = balanced_accuracy_score(test_labels, preds_base)
    test_acc_tuned = (preds_tuned == test_labels).mean()
    test_acc_base = (preds_base == test_labels).mean()

    pr, rc, f1 = precision_recall_fscore_support(test_labels, preds_tuned, labels=list(range(len(classes))), zero_division=0)[:3]
    cm = confusion_matrix(test_labels, preds_tuned)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics = {
        "best_threshold": best_thr,
        "rare_class": rare_class,
        "val_macro_f1": val_macro_f1,
        "val_bal_acc": val_bal,
        "test_macro_f1_tuned": test_macro_f1_tuned,
        "test_bal_acc_tuned": test_bal_tuned,
        "test_macro_f1_base": test_macro_f1_base,
        "test_bal_acc_base": test_bal_base,
        "test_acc_tuned": test_acc_tuned,
        "test_acc_base": test_acc_base,
        "per_class_precision": pr.tolist(),
        "per_class_recall": rc.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "config": saved_args,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Figures + explanations
    cm_path = out_dir / "confusion_matrix.png"
    sample_imgs = load_class_samples(sample_paths, input_size)
    plot_confusion(
        cm,
        classes,
        sample_imgs,
        cm_path,
        "Confusion matrix on the test set using the tuned rare-class threshold. Right column shows one sample image per class.",
    )

    per_class_path = out_dir / "per_class_metrics.png"
    plot_per_class(
        pr,
        rc,
        f1,
        classes,
        per_class_path,
        "Per-class precision/recall/F1 on the test set with the tuned threshold.",
    )

    thr_path = out_dir / "threshold_sweep.png"
    f1s, bals = [], []
    for thr in grid:
        preds = apply_threshold(val_probs, rare_idx, thr)
        f1s.append(f1_score(val_labels, preds, average="macro", zero_division=0))
        bals.append(balanced_accuracy_score(val_labels, preds))
    plot_threshold_sweep(
        grid,
        f1s,
        bals,
        rare_class,
        thr_path,
        "Validation sweep over rare-class thresholds showing macro F1 and balanced accuracy.",
    )

    print(f"Metrics saved to {out_dir / 'metrics.json'}")
    print(f"Figures saved to {cm_path}, {per_class_path}, {thr_path}")


if __name__ == "__main__":
    main()
