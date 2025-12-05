#!/usr/bin/env python3
"""
Train/evaluate the best 5 and worst 5 configs from a tuning summary.

Metrics: macro F1, balanced accuracy, confusion matrix (tuned threshold and baseline).
Decision threshold for the rarest class is tuned on the validation set via a grid search.
Outputs: JSON metrics and publication-style figures in the chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import optim
from torch.utils.data import DataLoader

import train_droplet as td


# Ensure matplotlib can write cache in restricted environments
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


@dataclass
class TrialConfig:
    trial_id: int
    config: Dict


@dataclass
class EvalResult:
    trial_id: int
    tag: str  # "best" or "worst"
    tuned_threshold: float
    rare_class: str
    rare_class_idx: int
    val_macro_f1: float
    val_bal_acc: float
    test_macro_f1_tuned: float
    test_bal_acc_tuned: float
    test_macro_f1_base: float
    test_bal_acc_base: float
    test_acc_tuned: float
    test_acc_base: float
    confusion_matrix: List[List[int]]
    config: Dict


def load_trials(summary_path: Path, k: int) -> Tuple[List[TrialConfig], List[TrialConfig]]:
    data = json.loads(summary_path.read_text())
    trials = data["trials"]
    ordered = sorted(trials, key=lambda t: t["metrics"]["test_acc"], reverse=True)
    best = [TrialConfig(t["trial_id"], t["config"]) for t in ordered[:k]]
    worst = [TrialConfig(t["trial_id"], t["config"]) for t in ordered[-k:]]
    return best, worst


def build_loaders_for_config(
    cfg: Dict,
    args: argparse.Namespace,
    device: torch.device,
    targets: List[int],
    train_idx: List[int],
):
    class_weights, sample_weights = td.compute_class_weights(train_idx, targets, cfg["num_classes"])
    train_loader, val_loader, test_loader, criterion = td.make_loaders(
        cfg["train_ds"],
        cfg["val_ds"],
        cfg["test_ds"],
        class_weights,
        sample_weights,
        cfg["batch_size"],
        args.num_workers,
        device,
        label_smoothing=cfg["label_smoothing"],
    )
    return train_loader, val_loader, test_loader, criterion


def collect_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            logits = model(images)
            p = softmax(logits).cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
    return np.vstack(probs), np.concatenate(labels)


def tune_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    rare_idx: int,
    grid: List[float],
) -> Tuple[float, float, float]:
    best_f1 = -1.0
    best_bal = -1.0
    best_thr = 0.5
    for thr in grid:
        preds = apply_threshold(probs, rare_idx, thr)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        bal = balanced_accuracy_score(labels, preds)
        if macro_f1 > best_f1 or (np.isclose(macro_f1, best_f1) and bal > best_bal):
            best_f1 = macro_f1
            best_bal = bal
            best_thr = thr
    return best_thr, best_f1, best_bal


def apply_threshold(probs: np.ndarray, rare_idx: int, threshold: float) -> np.ndarray:
    preds = probs.argmax(axis=1)
    rare_mask = probs[:, rare_idx] >= threshold
    preds[rare_mask] = rare_idx
    return preds


def plot_confusion(cm: np.ndarray, classes: List[str], title: str, out_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results: List[EvalResult], out_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = [f"T{r.trial_id} ({r.tag})" for r in results]
    macro_base = [r.test_macro_f1_base for r in results]
    macro_tuned = [r.test_macro_f1_tuned for r in results]
    bal_base = [r.test_bal_acc_base for r in results]
    bal_tuned = [r.test_bal_acc_tuned for r in results]

    x = np.arange(len(results))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(x - width / 2, macro_base, width, label="Macro F1 (base)", color="#95A5A6")
    axes[0].bar(x + width / 2, macro_tuned, width, label="Macro F1 (tuned)", color="#1ABC9C")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Macro F1: Base vs Tuned Threshold", fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(x - width / 2, bal_base, width, label="Balanced Acc (base)", color="#95A5A6")
    axes[1].bar(x + width / 2, bal_tuned, width, label="Balanced Acc (tuned)", color="#2980B9")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_title("Balanced Accuracy: Base vs Tuned Threshold", fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].legend(frameon=False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Best vs Worst Configs: Effect of Rare-Class Threshold Tuning", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(
    trial: TrialConfig,
    tag: str,
    args: argparse.Namespace,
    base_args: Dict,
    out_dir: Path,
    device: torch.device,
) -> EvalResult:
    cfg = dict(trial.config)

    td.seed_everything(base_args["seed"] + trial.trial_id)

    # Build datasets for this input size
    data_splits = td.build_datasets(
        Path(base_args["data_dir"]),
        input_size=cfg["input_size"],
        seed=base_args["seed"],
        val_split=base_args["val_split"],
        test_split=base_args["test_split"],
    )
    train_ds, val_ds, test_ds, classes, class_to_idx, targets, train_idx = data_splits
    cfg["num_classes"] = len(classes)

    # Identify rare class from training distribution
    counts = torch.bincount(torch.tensor([targets[i] for i in train_idx]), minlength=len(classes))
    rare_idx = int(counts.argmin().item())
    rare_class = classes[rare_idx]

    cfg["train_ds"] = train_ds
    cfg["val_ds"] = val_ds
    cfg["test_ds"] = test_ds

    train_loader, val_loader, test_loader, criterion = build_loaders_for_config(
        cfg, args, device, targets, train_idx
    )

    model, _ = td.build_model(len(classes), device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["sched_factor"], patience=cfg["sched_patience"]
    )

    best_state = None
    best_val = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = td.train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = td.evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(
            f"[{tag} trial {trial.trial_id}] epoch {epoch:02d} "
            f"train_loss={tr_loss:.3f} val_acc={val_acc:.3f}"
        )

    if best_state:
        model.load_state_dict(best_state)

    # Collect probabilities
    val_probs, val_labels = collect_probs(model, val_loader, device)
    test_probs, test_labels = collect_probs(model, test_loader, device)

    grid = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    best_thr, val_f1, val_bal = tune_threshold(val_probs, val_labels, rare_idx, list(grid))

    # Evaluate test with tuned threshold
    preds_tuned = apply_threshold(test_probs, rare_idx, best_thr)
    preds_base = test_probs.argmax(axis=1)
    cm = confusion_matrix(test_labels, preds_tuned).tolist()

    test_macro_f1_tuned = f1_score(test_labels, preds_tuned, average="macro", zero_division=0)
    test_bal_acc_tuned = balanced_accuracy_score(test_labels, preds_tuned)
    test_acc_tuned = (preds_tuned == test_labels).mean()

    test_macro_f1_base = f1_score(test_labels, preds_base, average="macro", zero_division=0)
    test_bal_acc_base = balanced_accuracy_score(test_labels, preds_base)
    test_acc_base = (preds_base == test_labels).mean()

    cm_path = out_dir / f"confusion_trial{trial.trial_id}_{tag}.png"
    plot_confusion(np.array(cm), classes, f"Trial {trial.trial_id} ({tag})", cm_path)

    return EvalResult(
        trial_id=trial.trial_id,
        tag=tag,
        tuned_threshold=float(best_thr),
        rare_class=rare_class,
        rare_class_idx=rare_idx,
        val_macro_f1=float(val_f1),
        val_bal_acc=float(val_bal),
        test_macro_f1_tuned=float(test_macro_f1_tuned),
        test_bal_acc_tuned=float(test_bal_acc_tuned),
        test_macro_f1_base=float(test_macro_f1_base),
        test_bal_acc_base=float(test_bal_acc_base),
        test_acc_tuned=float(test_acc_tuned),
        test_acc_base=float(test_acc_base),
        confusion_matrix=cm,
        config=trial.config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and analyze best/worst configs with threshold tuning.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to tuning_summary.json (defaults to newest under tuning_runs/).",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of best/worst configs to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_runs"), help="Where to write outputs.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=19, help="Number of steps in threshold grid.")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'.")
    return parser.parse_args()


def find_latest_summary() -> Path:
    candidates = sorted(Path("tuning_runs").glob("*/*tuning_summary.json"))
    if not candidates:
        raise FileNotFoundError("No tuning_summary.json found under tuning_runs/")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    args = parse_args()
    summary_path = args.summary or find_latest_summary()
    best, worst = load_trials(summary_path, args.topk)

    data = json.loads(summary_path.read_text())
    base_args = data.get("args", {})
    base_args.setdefault("seed", 123)
    base_args.setdefault("val_split", 0.15)
    base_args.setdefault("test_split", 0.15)
    base_args.setdefault("data_dir", "../matlab_env/CNN-for-droplet-sorting/Training Data")
    base_args["classes"] = None  # will be set per run

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[EvalResult] = []
    for tag, group in (("best", best), ("worst", worst)):
        for trial in group:
            res = train_and_evaluate(trial, tag, args, base_args, out_dir, device)
            results.append(res)

    # Sort results with best first then worst for plotting readability
    results = sorted(results, key=lambda r: (r.tag != "best", r.trial_id))

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    summary_fig = out_dir / "summary_metrics.png"
    plot_summary(results, summary_fig)

    print(f"Summary: {summary_fig}")
    print(f"Metrics JSON: {metrics_path}")
    print("Per-trial confusion matrices saved in output directory.")


if __name__ == "__main__":
    main()
