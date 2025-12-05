#!/usr/bin/env python3
"""Extended hyperparameter tuner for the droplet classifier.

Sweeps LR, weight decay, batch size, epochs, input size, scheduler params,
label smoothing. Optional wandb logging per trial.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.io import savemat
from torch import optim

import train_droplet as td


# --- helpers -----------------------------------------------------------------

def _safe_config_val(v):
    if isinstance(v, Path):
        return str(v)
    if v is None:
        return ""
    return v


def _mat_safe(val):
    if val is None:
        return ""
    if isinstance(val, Path):
        return str(val)
    return val


def _start_wandb_trial(base_args: argparse.Namespace, classes: List[str], cfg: Dict, trial_id: int):
    if not getattr(base_args, "wandb", False):
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb is not installed; skipping wandb logging.")
        return None

    api_key = base_args.wandb_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    base_cfg = {k: _safe_config_val(v) for k, v in vars(base_args).items()}
    base_cfg.update({f"trial_{k}": v for k, v in cfg.items()})
    base_cfg["num_classes"] = len(classes)
    base_cfg["classes"] = classes

    run = wandb.init(
        project=base_args.wandb_project,
        entity=base_args.wandb_entity or None,
        name=base_args.wandb_run_prefix + f"trial_{trial_id}" if base_args.wandb_run_prefix else f"trial_{trial_id}",
        group=base_args.wandb_group or "tuning_ext",
        job_type="tune_ext",
        config=base_cfg,
        reinit=False,
    )
    return run


# --- tuning core -------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_data = base_dir / "matlab_env" / "CNN-for-droplet-sorting" / "Training Data"
    default_out = base_dir / "tuning_runs"

    parser = argparse.ArgumentParser(description="Extended hyperparameter tuner for droplet CNN (Python).")
    parser.add_argument("--data-dir", type=Path, default=default_data, help="Folder with class subdirectories.")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Where to store tuning outputs.")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123, help="Base seed for reproducibility.")

    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 5e-4])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4, 5e-4])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--epochs-list", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--input-sizes", nargs="+", type=int, default=[64, 96])
    parser.add_argument("--sched-patiences", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--sched-factors", nargs="+", type=float, default=[0.5, 0.3])
    parser.add_argument("--label-smoothings", nargs="+", type=float, default=[0.0, 0.05])

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging for tuning trials.")
    parser.add_argument("--wandb-project", type=str, default="droplet-cnn", help="wandb project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/org.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Group name for this tuning run.")
    parser.add_argument("--wandb-run-prefix", type=str, default="", help="Prefix for trial run names.")
    parser.add_argument("--wandb-key", type=str, default=None, help="WANDB API key (or set WANDB_API_KEY env).")
    return parser.parse_args()


def run_trial(
    trial_id: int,
    cfg: Dict[str, float],
    device: torch.device,
    base_args: argparse.Namespace,
    wandb_run=None,
):
    # Build datasets for this input size and splits
    data_splits = td.build_datasets(
        base_args.data_dir,
        input_size=cfg["input_size"],
        seed=base_args.seed,
        val_split=base_args.val_split,
        test_split=base_args.test_split,
    )
    (
        train_ds,
        val_ds,
        test_ds,
        classes,
        class_to_idx,
        targets,
        train_idx,
    ) = data_splits

    td.seed_everything(base_args.seed + trial_id)

    class_weights, sample_weights = td.compute_class_weights(train_idx, targets, len(classes))
    model, norm = td.build_model(len(classes), device)
    train_loader, val_loader, test_loader, criterion = td.make_loaders(
        train_ds,
        val_ds,
        test_ds,
        class_weights,
        sample_weights,
        cfg["batch_size"],
        base_args.num_workers,
        device,
        label_smoothing=cfg["label_smoothing"],
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["sched_factor"], patience=cfg["sched_patience"]
    )

    best_state = None
    best_val = 0.0
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
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
        if wandb_run:
            import wandb  # type: ignore

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "trial_id": trial_id,
                }
            )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_acc = td.evaluate(model, test_loader, criterion, device)

    metrics = {
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }
    return {
        "trial_id": trial_id,
        "config": cfg,
        "metrics": metrics,
        "history": history,
        "model_state": best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()},
        "norm": norm,
        "classes": classes,
        "class_to_idx": class_to_idx,
    }


def main() -> None:
    args = parse_args()
    td.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) / f"tune_ext_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    grids = list(
        itertools.product(
            args.lrs,
            args.weight_decays,
            args.batch_sizes,
            args.epochs_list,
            args.input_sizes,
            args.sched_patiences,
            args.sched_factors,
            args.label_smoothings,
        )
    )

    trials: List[Dict] = []
    best = None
    for i, (lr, wd, bs, epochs, input_size, sched_patience, sched_factor, label_smoothing) in enumerate(grids, start=1):
        cfg = {
            "lr": lr,
            "weight_decay": wd,
            "batch_size": bs,
            "epochs": epochs,
            "input_size": input_size,
            "sched_patience": sched_patience,
            "sched_factor": sched_factor,
            "label_smoothing": label_smoothing,
        }
        print(
            f"[Trial {i}/{len(grids)}] lr={lr} wd={wd} bs={bs} epochs={epochs} size={input_size} "
            f"pat={sched_patience} fact={sched_factor} ls={label_smoothing}"
        )
        wandb_run = _start_wandb_trial(args, [], cfg, i)  # classes unknown yet; pass empty
        result = run_trial(i, cfg, device, args, wandb_run=wandb_run)
        if wandb_run:
            import wandb  # type: ignore

            wandb.config.update({"classes": result["classes"]}, allow_val_change=True)
            wandb.log(
                {
                    "best_val_acc": result["metrics"]["best_val_acc"],
                    "test_acc": result["metrics"]["test_acc"],
                    "test_loss": result["metrics"]["test_loss"],
                    "trial_id": i,
                }
            )
            wandb_run.finish()
        trials.append(
            {
                "trial_id": result["trial_id"],
                "config": cfg,
                "metrics": result["metrics"],
            }
        )
        if best is None or result["metrics"]["best_val_acc"] > best["metrics"]["best_val_acc"]:
            best = result

    if best is None:
        raise RuntimeError("No trials completed.")

    # Ensure JSON-safe values (convert Paths)
    json_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary = {
        "trials": trials,
        "best": {
            "trial_id": best["trial_id"],
            "config": best["config"],
            "metrics": best["metrics"],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": json_args,
    }
    (out_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2))

    best_config_obj = {_k: _mat_safe(_v) for _k, _v in summary["best"]["config"].items()}
    best_metrics_obj = {_k: _mat_safe(_v) for _k, _v in summary["best"]["metrics"].items()}
    args_obj = {_k: _mat_safe(_v) for _k, _v in summary["args"].items()}
    savemat(
        out_dir / "tuning_summary.mat",
        {
            "best_config": np.array([best_config_obj], dtype=object),
            "best_metrics": np.array([best_metrics_obj], dtype=object),
            "classes": np.array(best["classes"], dtype=object).reshape(-1, 1),
            "class_to_idx": np.array([best["class_to_idx"][c] for c in best["classes"]], dtype=np.int64).reshape(-1, 1),
            "args": np.array([args_obj], dtype=object),
        },
    )

    # Export best model artifacts using the training helper for MATLAB consumption.
    export_args = argparse.Namespace(**{**vars(args), **best["config"]})
    best_model, _ = td.build_model(len(best["classes"]), device=torch.device("cpu"))
    best_model.load_state_dict(best["model_state"])
    td.export_artifacts(
        best_model,
        out_dir,
        best["classes"],
        best["class_to_idx"],
        best["norm"],
        export_args,
        best["metrics"],
    )
    print(
        f"Tuning complete. Best trial {best['trial_id']} | val acc {best['metrics']['best_val_acc']:.3f} | "
        f"test acc {best['metrics']['test_acc']:.3f}"
    )
    print(f"Artifacts and summaries written to {out_dir}")


if __name__ == "__main__":
    main()
