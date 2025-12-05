import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


@dataclass
class Trial:
    trial_id: int
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    input_size: int
    sched_patience: int
    sched_factor: float
    label_smoothing: float
    best_val_acc: float
    test_acc: float
    test_loss: float

    @classmethod
    def from_dict(cls, entry: dict) -> "Trial":
        cfg = entry["config"]
        metrics = entry["metrics"]
        return cls(
            trial_id=entry["trial_id"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            input_size=cfg["input_size"],
            sched_patience=cfg["sched_patience"],
            sched_factor=cfg["sched_factor"],
            label_smoothing=cfg["label_smoothing"],
            best_val_acc=metrics["best_val_acc"],
            test_acc=metrics["test_acc"],
            test_loss=metrics["test_loss"],
        )


def load_trials(summary_path: pathlib.Path) -> List[Trial]:
    data = json.loads(summary_path.read_text())
    return [Trial.from_dict(t) for t in data["trials"]]


def summarize(trials: List[Trial], k: int = 5):
    ordered = sorted(trials, key=lambda t: t.test_acc, reverse=True)
    return ordered[:k], ordered[-k:]


def format_trials(trials: List[Trial]) -> str:
    rows = []
    for t in trials:
        rows.append(
            f"Trial {t.trial_id:3d} | acc={t.test_acc:.4f} | "
            f"val={t.best_val_acc:.4f} | loss={t.test_loss:.3f} | "
            f"lr={t.lr} wd={t.weight_decay} bs={t.batch_size} "
            f"ep={t.epochs} in={t.input_size} "
            f"pat={t.sched_patience} fac={t.sched_factor} "
            f"ls={t.label_smoothing}"
        )
    return "\n".join(rows)


def plot_bar(trials: List[Trial], title: str, out_path: pathlib.Path):
    labels = [f"T{t.trial_id}" for t in trials]
    accs = [t.test_acc for t in trials]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, accs, color="#2C3E50")
    ax.set_ylim(0, max(accs) * 1.05 if accs else 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_val_vs_test(trials: List[Trial], out_path: pathlib.Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(4, 4))
    vals = [t.best_val_acc for t in trials]
    tests = [t.test_acc for t in trials]
    sc = ax.scatter(vals, tests, s=20, alpha=0.6, edgecolor="#34495E", facecolor="#2980B9")
    ax.plot([0, 1], [0, 1], ls="--", color="#7F8C8D", linewidth=1)
    ax.set_xlabel("Best Validation Accuracy")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Validation vs. Test Accuracy", fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overview(trials: List[Trial], best: List[Trial], worst: List[Trial], out_path: pathlib.Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], width_ratios=[1.1, 1, 0.95], wspace=0.35, hspace=0.6)

    # Panel A: distribution of test accuracy
    ax_dist = fig.add_subplot(gs[0, :2])
    accs = [t.test_acc for t in trials]
    ax_dist.hist(accs, bins=20, color="#2C3E50", alpha=0.85, edgecolor="white")
    ax_dist.axvline(max(accs), color="#27AE60", linestyle="--", linewidth=1.2, label="Best")
    ax_dist.axvline(sum(accs) / len(accs), color="#8E44AD", linestyle=":", linewidth=1.2, label="Mean")
    ax_dist.set_xlabel("Test Accuracy")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Test Accuracy Distribution", fontweight="bold")
    ax_dist.set_xlim(0, 1.05)
    ax_dist.legend(frameon=False)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    # Panel B: validation vs test scatter, colored by label smoothing
    ax_scatter = fig.add_subplot(gs[0, 2])
    vals = [t.best_val_acc for t in trials]
    tests = [t.test_acc for t in trials]
    palette = {0.0: "#2980B9", 0.05: "#C0392B"}
    colors = [palette.get(t.label_smoothing, "#7F8C8D") for t in trials]
    ax_scatter.scatter(vals, tests, s=18, alpha=0.6, edgecolor="#34495E", facecolor=colors)
    ax_scatter.plot([0, 1], [0, 1], ls="--", color="#95A5A6", linewidth=1)
    ax_scatter.set_xlabel("Best Validation Accuracy")
    ax_scatter.set_ylabel("Test Accuracy")
    ax_scatter.set_title("Validation vs. Test", fontweight="bold")
    ax_scatter.set_xlim(0, 1.02)
    ax_scatter.set_ylim(0, 1.02)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    # Panel C: top 5 bar
    ax_top = fig.add_subplot(gs[1, :2])
    labels = [f"T{t.trial_id}" for t in best]
    accs_best = [t.test_acc for t in best]
    bars = ax_top.bar(labels, accs_best, color="#1ABC9C")
    ax_top.set_ylim(0, max(accs_best) * 1.05 if accs_best else 1)
    ax_top.set_ylabel("Test Accuracy")
    ax_top.set_title("Top 5 Trials", fontweight="bold")
    ax_top.tick_params(axis="x", rotation=0)
    for bar, acc in zip(bars, accs_best):
        ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Panel D: worst 5 horizontal bar
    ax_worst = fig.add_subplot(gs[1, 2])
    labels_w = [f"T{t.trial_id}" for t in reversed(worst)]
    accs_w = [t.test_acc for t in reversed(worst)]
    bars_w = ax_worst.barh(labels_w, accs_w, color="#E74C3C")
    ax_worst.set_xlim(0, max(accs) * 1.05 if accs else 1)
    ax_worst.set_xlabel("Test Accuracy")
    ax_worst.set_title("Bottom 5 Trials", fontweight="bold")
    for bar, acc in zip(bars_w, accs_w):
        ax_worst.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{acc:.3f}", va="center", fontsize=8)
    ax_worst.spines["top"].set_visible(False)
    ax_worst.spines["right"].set_visible(False)

    fig.suptitle("Tuning Overview", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_latest_summary() -> pathlib.Path:
    candidates = sorted(pathlib.Path("tuning_runs").glob("*/*tuning_summary.json"))
    if not candidates:
        raise FileNotFoundError("No tuning_summary.json found under tuning_runs/")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Summarize tuning runs.")
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        default=None,
        help="Path to tuning_summary.json (defaults to most recent under tuning_runs/).",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("figures"),
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of best/worst trials to report.",
    )
    args = parser.parse_args()

    summary_path = args.summary or find_latest_summary()
    trials = load_trials(summary_path)
    best, worst = summarize(trials, k=args.topk)

    args.outdir.mkdir(parents=True, exist_ok=True)
    best_fig = args.outdir / "best_topk.png"
    worst_fig = args.outdir / "worst_topk.png"
    scatter_fig = args.outdir / "val_vs_test.png"
    overview_fig = args.outdir / "tuning_overview.png"

    plot_bar(best, f"Top {args.topk} Trials (Test Acc)", best_fig)
    plot_bar(list(reversed(worst)), f"Worst {args.topk} Trials (Test Acc)", worst_fig)
    plot_val_vs_test(trials, scatter_fig)
    plot_overview(trials, best, worst, overview_fig)

    print(f"Summary path: {summary_path}")
    print("\nTop performers:")
    print(format_trials(best))
    print("\nLowest performers:")
    print(format_trials(worst))
    print("\nFigures saved to:")
    print(f"  {best_fig}")
    print(f"  {worst_fig}")
    print(f"  {scatter_fig}")
    print(f"  {overview_fig}")


if __name__ == "__main__":
    main()
