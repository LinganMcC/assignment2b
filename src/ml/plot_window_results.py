"""
plot_window_results.py
======================
Standalone plot generator for the window-size experiment.
Reads the existing results.json — no retraining required.

Usage (from project root):
    python -m src.ml.plot_window_results
    python -m src.ml.plot_window_results --results-path path/to/results.json
    python -m src.ml.plot_window_results --out-dir path/to/plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── default paths (mirror experiment_window_size.py) ─────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS = (
    _PROJECT_ROOT
    / "data" / "processed" / "ml_artefacts" / "window_experiment" / "results.json"
)
_DEFAULT_PLOTS = _DEFAULT_RESULTS.parent / "plots"

MODEL_COLOURS = {
    "lstm":        "#2196F3",
    "gru":         "#4CAF50",
    "transformer": "#FF5722",
}
MARKERS = {"lstm": "o", "gru": "s", "transformer": "^"}
MODEL_ORDER = ["lstm", "gru", "transformer"]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load(results_path: Path) -> list[dict]:
    with open(results_path) as f:
        return json.load(f)


def _series(results: list[dict], model: str, metric: str,
            windows: list[int]) -> list[float]:
    lookup = {r["window_size"]: r[metric]
              for r in results if r["model"] == model}
    return [lookup.get(w, float("nan")) for w in windows]


def _xtick_labels(windows: list[int]) -> list[str]:
    return [f"{w}\n({w * 15} min)" for w in windows]


# ═══════════════════════════════════════════════════════════════════════════
# Individual plot functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_line(plt, results, windows, metric, ylabel, title, fname, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in MODEL_ORDER:
        ys = _series(results, m, metric, windows)
        ax.plot(windows, ys,
                color=MODEL_COLOURS[m], marker=MARKERS[m],
                linewidth=1.8, markersize=6, label=m.upper())
        best_idx = int(np.nanargmin(ys))
        ax.annotate(
            f"{ys[best_idx]:.4g}",
            xy=(windows[best_idx], ys[best_idx]),
            xytext=(4, -13), textcoords="offset points",
            fontsize=7.5, color=MODEL_COLOURS[m],
        )
    ax.set_xlabel("Window size (steps × 15 min)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(windows)
    ax.set_xticklabels(_xtick_labels(windows), fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_epochs(plt, results, windows, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in MODEL_ORDER:
        ys = _series(results, m, "epochs_run", windows)
        ax.plot(windows, ys,
                color=MODEL_COLOURS[m], marker=MARKERS[m],
                linewidth=1.8, markersize=6, label=m.upper())
    ax.set_xlabel("Window size (steps × 15 min)", fontsize=10)
    ax.set_ylabel("Epochs until early stop", fontsize=10)
    ax.set_title("Training Epochs vs Window Size", fontsize=11, fontweight="bold")
    ax.set_xticks(windows)
    ax.set_xticklabels(_xtick_labels(windows), fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "training_epochs_vs_window.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_train_time(plt, results, windows, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in MODEL_ORDER:
        ys = [s / 60 for s in _series(results, m, "train_sec", windows)]
        ax.plot(windows, ys,
                color=MODEL_COLOURS[m], marker=MARKERS[m],
                linewidth=1.8, markersize=6, label=m.upper())
    ax.set_xlabel("Window size (steps × 15 min)", fontsize=10)
    ax.set_ylabel("Wall-clock training time (minutes)", fontsize=10)
    ax.set_title("Training Time vs Window Size", fontsize=11, fontweight="bold")
    ax.set_xticks(windows)
    ax.set_xticklabels(_xtick_labels(windows), fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "training_time_vs_window.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_heatmap(plt, results, windows, out_dir):
    data = np.array([
        [_series(results, m, "MAE", windows)[wi] for wi in range(len(windows))]
        for m in MODEL_ORDER
    ])
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"W={w}" for w in windows], fontsize=9)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([m.upper() for m in MODEL_ORDER], fontsize=9)
    ax.set_title("Test MAE (normalised) — Model × Window Size",
                 fontsize=11, fontweight="bold")
    median = np.nanmedian(data)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(windows)):
            ax.text(j, i, f"{data[i, j]:.4f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > median else "black")
    plt.colorbar(im, ax=ax, label="MAE (norm)")
    fig.tight_layout()
    out = out_dir / "heatmap_mae_norm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_delta_vs_baseline(plt, results, windows, out_dir):
    """Show change in MAE_veh relative to W=12 (the production baseline)."""
    BASELINE_W = 12
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Baseline (W=12)")
    for m in MODEL_ORDER:
        ys    = _series(results, m, "MAE_veh", windows)
        b_idx = windows.index(BASELINE_W) if BASELINE_W in windows else None
        if b_idx is None:
            print(f"  W={BASELINE_W} not in results — skipping delta plot")
            plt.close(fig)
            return
        baseline = ys[b_idx]
        deltas = [y - baseline for y in ys]
        ax.plot(windows, deltas,
                color=MODEL_COLOURS[m], marker=MARKERS[m],
                linewidth=1.8, markersize=6, label=m.upper())
    ax.set_xlabel("Window size (steps × 15 min)", fontsize=10)
    ax.set_ylabel("ΔMAE_veh vs W=12 (vehicles / 15 min)", fontsize=10)
    ax.set_title("MAE Change Relative to Production Window (W=12)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(windows)
    ax.set_xticklabels(_xtick_labels(windows), fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "delta_mae_vs_baseline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_best_per_model(plt, results, out_dir):
    best = {}
    for r in results:
        m = r["model"]
        if m not in best or r["MAE"] < best[m]["MAE"]:
            best[m] = r

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(MODEL_ORDER))
    vals = [best[m]["MAE_veh"] for m in MODEL_ORDER]
    bars = ax.bar(xs, vals,
                  color=[MODEL_COLOURS[m] for m in MODEL_ORDER],
                  edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f veh", padding=3, fontsize=9)
    for i, m in enumerate(MODEL_ORDER):
        w = best[m]["window_size"]
        ax.text(i, -0.002, f"W={w} ({w*15}m)",
                ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(xs)
    ax.set_xticklabels([m.upper() for m in MODEL_ORDER], fontsize=10)
    ax.set_ylabel("Best Test MAE (vehicles / 15 min)", fontsize=10)
    ax.set_title("Best MAE per Model (optimal window size labelled)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "best_mae_per_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_combined_overview(plt, results, windows, out_dir):
    """2×2 summary figure suitable for the report."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Window-Size Sensitivity Analysis", fontsize=13, fontweight="bold")

    metrics = [
        (axes[0, 0], "MAE",     "MAE (normalised)",         "Test MAE"),
        (axes[0, 1], "MAE_veh", "MAE (vehicles / 15 min)",  "Test MAE (vehicles)"),
        (axes[1, 0], "RMSE",    "RMSE (normalised)",        "Test RMSE"),
        (axes[1, 1], "MAPE_%",  "MAPE (%)",                 "Test MAPE"),
    ]
    for ax, metric, ylabel, title in metrics:
        for m in MODEL_ORDER:
            ys = _series(results, m, metric, windows)
            ax.plot(windows, ys,
                    color=MODEL_COLOURS[m], marker=MARKERS[m],
                    linewidth=1.6, markersize=5, label=m.upper())
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Window size (steps)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(windows)
        ax.set_xticklabels([str(w) for w in windows], fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "overview_2x2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(results_path: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load(results_path)
    windows = sorted(set(r["window_size"] for r in results))
    print(f"Loaded {len(results)} result rows — "
          f"windows={windows}, models={list(MODEL_REGISTRY_ORDER := MODEL_ORDER)}")
    print(f"Writing plots to {out_dir}\n")

    # ── line charts ──────────────────────────────────────────────────────
    plot_line(plt, results, windows,
              "MAE",     "MAE (normalised)",        "Test MAE vs Window Size",
              "mae_vs_window.png", out_dir)

    plot_line(plt, results, windows,
              "RMSE",    "RMSE (normalised)",       "Test RMSE vs Window Size",
              "rmse_vs_window.png", out_dir)

    plot_line(plt, results, windows,
              "MAPE_%",  "MAPE (%)",                "Test MAPE vs Window Size",
              "mape_vs_window.png", out_dir)

    plot_line(plt, results, windows,
              "MAE_veh", "MAE (vehicles / 15 min)", "Test MAE (vehicles) vs Window Size",
              "mae_veh_vs_window.png", out_dir)

    # ── convergence & cost ───────────────────────────────────────────────
    plot_epochs(plt, results, windows, out_dir)
    plot_train_time(plt, results, windows, out_dir)

    # ── comparison / insight plots ───────────────────────────────────────
    plot_heatmap(plt, results, windows, out_dir)
    plot_delta_vs_baseline(plt, results, windows, out_dir)
    plot_best_per_model(plt, results, out_dir)
    plot_combined_overview(plt, results, windows, out_dir)

    print(f"\nDone — {len(list(out_dir.glob('*.png')))} plots saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from window-size experiment results"
    )
    parser.add_argument(
        "--results-path", type=Path, default=_DEFAULT_RESULTS,
        help="Path to results.json (default: ml_artefacts/window_experiment/results.json)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_PLOTS,
        help="Output directory for plots (default: …/window_experiment/plots/)",
    )
    args = parser.parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(
            f"results.json not found at {args.results_path}\n"
            "Run experiment_window_size.py first."
        )

    main(args.results_path, args.out_dir)