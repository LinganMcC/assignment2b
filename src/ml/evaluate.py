
from __future__ import annotations

import json


import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_datasets, ARTEFACTS_DIR
from models import build_model, MODEL_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTS_DIR = ARTEFACTS_DIR / "plots"




def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    # MAPE — guard against zero targets
    nonzero = targets != 0
    mape = float(np.mean(np.abs((preds[nonzero] - targets[nonzero]) / targets[nonzero])) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2)}


def run_inference(model_name: str, datasets: dict, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds_norm, targets_norm) arrays for the test split."""
    ckpt = ARTEFACTS_DIR / f"{model_name}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run train.py first.")

    model = build_model(model_name, n_features=datasets["n_features"]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(datasets["X_test"]),
                      torch.from_numpy(datasets["y_test"]).unsqueeze(1)),
        batch_size=batch_size, shuffle=False,
    )
    preds, tgts = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.append(model(X_b.to(DEVICE)).cpu().numpy())
            tgts.append(y_b.numpy())

    return np.concatenate(preds).squeeze(), np.concatenate(tgts).squeeze()




def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return None


def plot_training_curves(plt, histories: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colours = {"lstm": "#2196F3", "gru": "#4CAF50", "transformer": "#FF5722"}

    for name, hist in histories.items():
        c = colours.get(name, "grey")
        axes[0].plot(hist["train_loss"], label=f"{name} train", color=c, linestyle="--", alpha=0.7)
        axes[0].plot(hist["val_loss"],   label=f"{name} val",   color=c, linestyle="-")
        axes[1].plot(hist["val_mae"],    label=f"{name} val",   color=c, linestyle="-")

    for ax, title, ylabel in zip(
        axes,
        ["Training & Validation Loss (MSE)", "Validation MAE"],
        ["MSE", "MAE (normalised)"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_prediction_sample(
    plt,
    datasets: dict,
    preds_by_model: dict[str, np.ndarray],
    n_intervals: int = 96,
) -> None:
    """Plot actual vs predicted for a single site on a single test day."""
    site_ids = datasets["sites_test"]
    unique_sites = np.unique(site_ids)
    rng = np.random.default_rng(42)
    chosen_site = int(rng.choice(unique_sites))

    mask = site_ids == chosen_site
    actual = datasets["y_test"][mask][:n_intervals]

    if len(actual) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(actual))
    ax.plot(x, actual, label="Actual", color="black", linewidth=2)

    colours = {"lstm": "#2196F3", "gru": "#4CAF50", "transformer": "#FF5722"}
    for name, preds in preds_by_model.items():
        site_preds = preds[mask][:n_intervals]
        ax.plot(x, site_preds, label=name.upper(),
                color=colours.get(name, "grey"), alpha=0.8, linewidth=1.2)

    ax.set_xlabel("15-min interval")
    ax.set_ylabel("Flow (normalised)")
    ax.set_title(f"Actual vs Predicted — Site {chosen_site} (first {n_intervals} test intervals)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "prediction_sample.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_model_comparison(plt, results: list[dict]) -> None:
    """Bar chart comparing test MAE (vehicles) across models."""
    names = [r["model"].upper() for r in results]
    maes  = [r["test_MAE_veh"] for r in results]
    colours = ["#2196F3", "#4CAF50", "#FF5722"][: len(names)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, maes, color=colours, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    ax.set_ylabel("Test MAE (vehicles / 15 min)")
    ax.set_title("Model Comparison — Test Set MAE")
    ax.set_ylim(0, max(maes) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")



def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets ...")
    datasets = load_datasets(save_scaler=False)   # scaler already saved by train.py

    all_results: list[dict] = []
    preds_by_model: dict[str, np.ndarray] = {}
    histories: dict[str, dict] = {}

    for name in MODEL_REGISTRY:
        print(f"\nEvaluating {name.upper()} ...")
        try:
            preds, targets = run_inference(name, datasets)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Metrics in normalised space
        norm_metrics = compute_metrics(preds, targets)

        # De-normalise for interpretable MAE
        scaler = datasets["scaler"]
        site_ids = datasets["sites_test"]
        denorm_p = np.array([scaler.inverse_transform(int(s), np.array([p]))[0]
                              for s, p in zip(site_ids, preds)])
        denorm_t = np.array([scaler.inverse_transform(int(s), np.array([t]))[0]
                              for s, t in zip(site_ids, targets)])
        veh_metrics = compute_metrics(denorm_p, denorm_t)

        result = {
            "model": name,
            "test_MAE":      norm_metrics["MAE"],
            "test_RMSE":     norm_metrics["RMSE"],
            "test_MAPE_%":   norm_metrics["MAPE_%"],
            "test_MAE_veh":  veh_metrics["MAE"],
            "test_RMSE_veh": veh_metrics["RMSE"],
        }
        all_results.append(result)
        preds_by_model[name] = preds

        print(f"  MAE (norm)={norm_metrics['MAE']:.4f}  "
              f"RMSE (norm)={norm_metrics['RMSE']:.4f}  "
              f"MAPE={norm_metrics['MAPE_%']:.1f}%  "
              f"MAE (veh)={veh_metrics['MAE']:.1f}")

        # Load training history if available
        hist_path = ARTEFACTS_DIR / f"{name}_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                histories[name] = json.load(f)

    # Save results
    out_path = ARTEFACTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Comparison table
    print(f"\n{'─'*65}")
    print(f"{'Model':<14} {'MAE':>8} {'RMSE':>8} {'MAPE%':>7} {'MAE(veh)':>10} {'RMSE(veh)':>10}")
    print("─" * 65)
    for r in all_results:
        print(f"{r['model']:<14} {r['test_MAE']:>8.4f} {r['test_RMSE']:>8.4f} "
              f"{r['test_MAPE_%']:>6.1f}% {r['test_MAE_veh']:>10.1f} {r['test_RMSE_veh']:>10.1f}")
    print("─" * 65)

    # Plots
    plt = _try_import_matplotlib()
    if plt and all_results:
        print("\nGenerating plots ...")
        if histories:
            plot_training_curves(plt, histories)
        plot_prediction_sample(plt, datasets, preds_by_model)
        plot_model_comparison(plt, all_results)


if __name__ == "__main__":
    main()
