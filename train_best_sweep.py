#!/usr/bin/env python3
"""
Train the best tuning config with several epoch counts and save the best result.

Defaults to the latest tuning_summary.json under tuning_runs/, extracts the best
config, retrains with a list of epoch counts, and saves the best-performing model
and metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import optim

import train_droplet as td


def find_latest_summary() -> Path:
    candidates = sorted(Path("tuning_runs").glob("*/*tuning_summary.json"))
    if not candidates:
        raise FileNotFoundError("No tuning_summary.json found under tuning_runs/")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_best_config(summary_path: Path) -> Dict:
    data = json.loads(summary_path.read_text())
    best = data["best"]["config"]
    args = data.get("args", {})
    return best, args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain best tuned config with multiple epoch counts.")
    parser.add_argument("--summary", type=Path, default=None, help="Path to tuning_summary.json (default: latest).")
    parser.add_argument("--epochs-list", nargs="+", type=int, default=[10, 15, 20], help="Epoch counts to try.")
    parser.add_argument("--output-dir", type=Path, default=Path("best_runs"), help="Where to save artifacts.")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    return parser.parse_args()


def build_datasets(cfg: Dict, args: Dict) -> Tuple:
    data_dir = Path(args.get("data_dir") or "../matlab_env/CNN-for-droplet-sorting/Training Data")
    val_split = float(args.get("val_split", 0.15))
    test_split = float(args.get("test_split", 0.15))
    seed = int(args.get("seed", 123))
    return td.build_datasets(
        data_dir,
        input_size=cfg["input_size"],
        seed=seed,
        val_split=val_split,
        test_split=test_split,
    )


def train_for_epochs(
    cfg: Dict,
    base_args: Dict,
    epochs: int,
    device: torch.device,
    num_workers: int,
) -> Dict:
    td.seed_everything(int(base_args.get("seed", 123)) + epochs)
    splits = build_datasets(cfg, base_args)
    train_ds, val_ds, test_ds, classes, class_to_idx, targets, train_idx = splits

    class_weights, sample_weights = td.compute_class_weights(train_idx, targets, len(classes))
    train_loader, val_loader, test_loader, criterion = td.make_loaders(
        train_ds,
        val_ds,
        test_ds,
        class_weights,
        sample_weights,
        cfg["batch_size"],
        num_workers,
        device,
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )

    model, norm = td.build_model(len(classes), device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["sched_factor"], patience=cfg["sched_patience"]
    )

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = td.train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = td.evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"[epochs={epochs}] epoch {epoch:02d} train_loss={tr_loss:.4f} val_acc={val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    test_loss, test_acc = td.evaluate(model, test_loader, criterion, device)
    return {
        "model": model,
        "norm": norm,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "history": history,
        "best_val_acc": best_val,
        "test_acc": test_acc,
    }


def main():
    args_ns = parse_args()
    summary_path = args_ns.summary or find_latest_summary()
    best_cfg, base_args = load_best_config(summary_path)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args_ns.device == "auto"
        else torch.device(args_ns.device)
    )
    num_workers = args_ns.num_workers if args_ns.num_workers is not None else int(base_args.get("num_workers", 4))
    seed_override = args_ns.seed
    if seed_override is not None:
        base_args["seed"] = seed_override

    out_dir = args_ns.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best_overall = None
    for ep in args_ns.epochs_list:
        cfg = dict(best_cfg)
        cfg["epochs"] = ep
        run = train_for_epochs(cfg, base_args, ep, device, num_workers)
        results.append(
            {
                "epochs": ep,
                "best_val_acc": run["best_val_acc"],
                "test_acc": run["test_acc"],
                "history": run["history"],
            }
        )
        if best_overall is None or run["best_val_acc"] > best_overall["best_val_acc"]:
            best_overall = {"epochs": ep, **run}
            best_overall["config"] = cfg

    # Save best model/artifacts
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_dir = out_dir / f"best_epochs_{best_overall['epochs']}_{timestamp}"
    best_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "best_val_acc": best_overall["best_val_acc"],
        "test_acc": best_overall["test_acc"],
        "epochs": best_overall["epochs"],
        "config": best_overall["config"],
    }
    td.export_artifacts(
        best_overall["model"],
        best_dir,
        best_overall["classes"],
        best_overall["class_to_idx"],
        best_overall["norm"],
        argparse.Namespace(**best_overall["config"]),
        metrics,
    )

    (out_dir / "sweep_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved best model to {best_dir}")
    print(f"Metrics: {out_dir / 'best_metrics.json'}")


if __name__ == "__main__":
    main()
