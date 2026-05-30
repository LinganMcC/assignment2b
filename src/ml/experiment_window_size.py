"""
experiment_window_size.py
=========================
Research component: "How does sliding window size affect traffic flow
prediction performance for LSTM, GRU, and Transformer models?"

Experiment design
-----------------
For each window size in WINDOW_SIZES:
  1. Re-generate train/val/test windows from the *same* normalised splits
     (scaler is fit once on training rows and reused).
  2. Re-train each model from a fixed random seed with identical
     hyper-parameters to the main pipeline.
  3. Evaluate on the held-out test split (days 28-31).
  4. Record MAE (normalised), RMSE (normalised), MAPE (%), MAE (vehicles),
     training epochs used, and wall-clock training time.

All results are written to:
    data/processed/ml_artefacts/window_experiment/
        results.json            — machine-readable summary
        results.csv             — spreadsheet-friendly summary
        plots/
            mae_vs_window.png
            rmse_vs_window.png
            mape_vs_window.png
            mae_veh_vs_window.png
            training_epochs_vs_window.png
            heatmap_mae_norm.png

Usage (from project root):
    python -m src.ml.experiment_window_size
    python -m src.ml.experiment_window_size --window-sizes 4 8 12 24 48
    python -m src.ml.experiment_window_size --epochs 30 --no-plots
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── project imports (same package) ──────────────────────────────────────────
from dataset import load_datasets, ARTEFACTS_DIR, TIMESERIES_CSV
from models import build_model, MODEL_REGISTRY


# ── experiment constants ─────────────────────────────────────────────────────
WINDOW_SIZES: list[int] = [4, 8, 12, 16, 24, 32, 48]

SEED = 42

# Training hyper-parameters — intentionally identical to train.py defaults
TRAIN_CFG = {
    "batch_size": 256,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 7,  # early-stopping patience on val MAE
    "clip_grad": 1.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXP_DIR = ARTEFACTS_DIR / "window_experiment"
PLOTS_DIR = EXP_DIR / "plots"

# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility helpers
# ═══════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ═══════════════════════════════════════════════════════════════════════════
# Data loader factory
# ═══════════════════════════════════════════════════════════════════════════

def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).unsqueeze(1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))

# ═══════════════════════════════════════════════════════════════════════════
# Single training run
# ═══════════════════════════════════════════════════════════════════════════

def _train_one(
        model_name: str,
        datasets: dict,
        cfg: dict,
        ckpt_path: Path,
) -> dict:
    """Train model_name on datasets, save best checkpoint.

    Returns history dict with train/val loss+MAE lists and epochs_run.
    """
    _set_seed(SEED)

    train_loader = _make_loader(datasets["X_train"], datasets["y_train"],
                                cfg["batch_size"], shuffle=True)
    val_loader = _make_loader(datasets["X_val"], datasets["y_val"],
                              cfg["batch_size"], shuffle=False)

    model = build_model(model_name,
                        n_features=datasets["n_features"]).to(DEVICE)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    patience_counter = 0
    epochs_run = 0

    for epoch in range(1, cfg["epochs"] + 1):
        epochs_run = epoch

        # ── train ──
        model.train()
        tr_loss, n = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            optimiser.step()
            tr_loss += loss.item();
            n += 1
        tr_loss /= n

        # ── validate ──
        model.eval()
        va_loss, va_mae, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(X_b)
                va_loss += criterion(pred, y_b).item()
                va_mae += (pred - y_b).abs().mean().item()
                nv += 1
        va_loss /= nv
        va_mae /= nv

        scheduler.step(va_loss)
        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(va_loss, 6))
        history["val_mae"].append(round(va_mae, 6))

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                break

    history["epochs_run"] = epochs_run
    return history

# ═══════════════════════════════════════════════════════════════════════════
# Inference + metrics
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate(
        model_name: str,
        datasets: dict,
        ckpt_path: Path,
        batch_size: int = 512,
) -> dict:
    """Load checkpoint, run on test set, return metric dict."""
    model = build_model(model_name,
                        n_features=datasets["n_features"]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    loader = _make_loader(datasets["X_test"], datasets["y_test"],
                          batch_size, shuffle=False)
    preds, tgts = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.append(model(X_b.to(DEVICE)).cpu().numpy())
            tgts.append(y_b.numpy())

    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(tgts).squeeze()

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    nz = targets != 0
    mape = float(np.mean(np.abs((preds[nz] - targets[nz]) / targets[nz])) * 100)

    # De-normalise → vehicle counts
    scaler = datasets["scaler"]
    site_ids = datasets["sites_test"]
    dp = np.array([scaler.inverse_transform(int(s), np.array([p]))[0]
                   for s, p in zip(site_ids, preds)])
    dt = np.array([scaler.inverse_transform(int(s), np.array([t]))[0]
                   for s, t in zip(site_ids, targets)])
    mae_veh = float(np.mean(np.abs(dp - dt)))
    rmse_veh = float(np.sqrt(np.mean((dp - dt) ** 2)))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_%": round(mape, 2),
        "MAE_veh": round(mae_veh, 2),
        "RMSE_veh": round(rmse_veh, 2),
    }

# ═══════════════════════════════════════════════════════════════════════════
# Main experiment loop
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
        window_sizes: list[int] = WINDOW_SIZES,
        cfg: dict = TRAIN_CFG,
        make_plots: bool = True,
) -> list[dict]:
    """Run the full grid: window_sizes × models.

    Returns list of result dicts, one per (window_size, model) pair.
    """
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load CSV once; normalisation is the same for every window size ──
    print("=" * 65)
    print("  Window-Size Experiment")
    print(f"  Windows : {window_sizes}")
    print(f"  Models  : {list(MODEL_REGISTRY)}")
    print(f"  Device  : {DEVICE}")
    print("=" * 65)

    # Load with default window_size just to get the normalised dataframes;
    # we'll rebuild windows per experiment cell below.
    # We call load_datasets with save_scaler=True once, then reuse the
    # returned scaler object (no re-fitting).
    base = load_datasets(window_size=12, save_scaler=True)
    scaler = base["scaler"]

    # We need the raw (pre-windowed) normalised splits.  Re-derive them
    # by reloading the CSV at the dataset level — but without re-fitting
    # the scaler.  The cleanest approach is to use load_datasets with each
    # window_size (it re-uses the fit call internally, but fit is
    # deterministic given the same training rows, so results are identical).
    # We cache the normalised DataFrames to avoid re-reading the CSV each time.

    import pandas as pd
    from dataset import add_time_features, TRAIN_END, VAL_END, build_windows, N_FEATURES

    print("\nCaching normalised splits ...")
    raw_df = pd.read_csv(TIMESERIES_CSV)
    raw_df = add_time_features(raw_df)
    raw_df["_day"] = pd.to_datetime(raw_df["datetime"]).dt.day

    train_df = scaler.transform(raw_df[raw_df["_day"] <= TRAIN_END].copy())
    val_df = scaler.transform(raw_df[(raw_df["_day"] > TRAIN_END) &
                                     (raw_df["_day"] <= VAL_END)].copy())
    test_df = scaler.transform(raw_df[raw_df["_day"] > VAL_END].copy())
    print("  Done.\n")

    all_results: list[dict] = []

    for ws in window_sizes:
        print(f"\n{'─' * 65}")
        print(f"  WINDOW SIZE = {ws}  ({ws * 15} min look-back)")
        print(f"{'─' * 65}")

        # Re-build windows for this window size
        X_tr, y_tr, s_tr = build_windows(train_df, ws)
        X_va, y_va, s_va = build_windows(val_df, ws)
        X_te, y_te, s_te = build_windows(test_df, ws)
        print(f"  Shapes — train:{X_tr.shape}  val:{X_va.shape}  test:{X_te.shape}")

        datasets = {
            "X_train": X_tr, "y_train": y_tr, "sites_train": s_tr,
            "X_val": X_va, "y_val": y_va, "sites_val": s_va,
            "X_test": X_te, "y_test": y_te, "sites_test": s_te,
            "scaler": scaler,
            "n_features": N_FEATURES,
        }

        for model_name in MODEL_REGISTRY:
            print(f"\n  [{model_name.upper()}]  window={ws}")

            ckpt = EXP_DIR / f"{model_name}_w{ws}_best.pt"

            t0 = time.time()
            history = _train_one(model_name, datasets, cfg, ckpt)
            train_sec = round(time.time() - t0, 1)

            metrics = _evaluate(model_name, datasets, ckpt)

            record = {
                "window_size": ws,
                "lookback_min": ws * 15,
                "model": model_name,
                "epochs_run": history["epochs_run"],
                "train_sec": train_sec,
                **metrics,
            }
            all_results.append(record)

            print(f"    epochs={history['epochs_run']}  "
                  f"MAE(norm)={metrics['MAE']:.4f}  "
                  f"RMSE(norm)={metrics['RMSE']:.4f}  "
                  f"MAPE={metrics['MAPE_%']:.1f}%  "
                  f"MAE(veh)={metrics['MAE_veh']:.1f}  "
                  f"time={train_sec}s")

    # ── persist results ──────────────────────────────────────────────────
    _save_results(all_results)

    if make_plots:
        _make_plots(all_results)

    _print_summary(all_results)

    return all_results

# ═══════════════════════════════════════════════════════════════════════════
# Persistence helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_results(results: list[dict]) -> None:
    # JSON
    json_path = EXP_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # CSV (human-friendly)
    import csv
    csv_path = EXP_DIR / "results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
    print(f"  Results saved → {csv_path}")

# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

MODEL_COLOURS = {
    "lstm": "#2196F3",
    "gru": "#4CAF50",
    "transformer": "#FF5722",
}
MARKERS = {"lstm": "o", "gru": "s", "transformer": "^"}

def _make_plots(results: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    # Organise data: { model_name: { metric: [values in window_size order] } }
    windows = sorted(set(r["window_size"] for r in results))
    models = list(MODEL_REGISTRY)

    def _series(model: str, metric: str) -> list[float]:
        lookup = {r["window_size"]: r[metric]
                  for r in results if r["model"] == model}
        return [lookup.get(w, float("nan")) for w in windows]

    # ── 1. Line charts for MAE, RMSE, MAPE, MAE_veh ──────────────────────
    metric_meta = [
        ("MAE", "MAE (normalised)", "Test MAE vs Window Size"),
        ("RMSE", "RMSE (normalised)", "Test RMSE vs Window Size"),
        ("MAPE_%", "MAPE (%)", "Test MAPE vs Window Size"),
        ("MAE_veh", "MAE (vehicles / 15 min)", "Test MAE (vehicles) vs Window Size"),
    ]

    for metric, ylabel, title in metric_meta:
        fig, ax = plt.subplots(figsize=(8, 4))
        for m in models:
            ys = _series(m, metric)
            ax.plot(windows, ys,
                    color=MODEL_COLOURS[m], marker=MARKERS[m],
                    linewidth=1.8, markersize=6, label=m.upper())
            # annotate the minimum
            best_idx = int(np.nanargmin(ys))
            ax.annotate(
                f"{ys[best_idx]:.3g}",
                xy=(windows[best_idx], ys[best_idx]),
                xytext=(4, -12), textcoords="offset points",
                fontsize=7, color=MODEL_COLOURS[m],
            )
        ax.set_xlabel("Window size (steps × 15 min)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(windows)
        ax.set_xticklabels([f"{w}\n({w * 15}m)" for w in windows], fontsize=7)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = metric.lower().replace("%", "pct").replace("_", "") + "_vs_window.png"
        out = PLOTS_DIR / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")

    # ── 2. Training epochs used (proxy for convergence speed) ─────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in models:
        ys = _series(m, "epochs_run")
        ax.plot(windows, ys,
                color=MODEL_COLOURS[m], marker=MARKERS[m],
                linewidth=1.8, markersize=6, label=m.upper())
    ax.set_xlabel("Window size (steps)")
    ax.set_ylabel("Epochs until early stop")
    ax.set_title("Training Epochs vs Window Size")
    ax.set_xticks(windows)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "training_epochs_vs_window.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

    # ── 3. Heat-map: MAE_norm[model × window_size] ────────────────────────
    try:
        import matplotlib.colors as mcolors

        data = np.array([[_series(m, "MAE")[wi] for wi in range(len(windows))]
                         for m in models])  # (n_models, n_windows)

        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(data, aspect="auto",
                       cmap="RdYlGn_r",
                       vmin=np.nanmin(data), vmax=np.nanmax(data))
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels([f"W={w}" for w in windows])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.upper() for m in models])
        ax.set_title("Test MAE (normalised) — Model × Window Size")
        for i in range(len(models)):
            for j in range(len(windows)):
                ax.text(j, i, f"{data[i, j]:.4f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > np.nanmedian(data) else "black")
        plt.colorbar(im, ax=ax, label="MAE (norm)")
        fig.tight_layout()
        out = PLOTS_DIR / "heatmap_mae_norm.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")
    except Exception as e:
        print(f"  Heatmap skipped: {e}")

    # ── 4. Bar chart: best window per model ───────────────────────────────
    best_rows = {}
    for r in results:
        key = r["model"]
        if key not in best_rows or r["MAE"] < best_rows[key]["MAE"]:
            best_rows[key] = r

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(models))
    bars = ax.bar(
        xs,
        [best_rows[m]["MAE_veh"] for m in models],
        color=[MODEL_COLOURS[m] for m in models],
        edgecolor="white", width=0.5,
    )
    ax.bar_label(bars, fmt="%.1f veh", padding=3, fontsize=9)
    for i, m in enumerate(models):
        w = best_rows[m]["window_size"]
        ax.text(i, -0.004, f"W={w}", ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(xs)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Best Test MAE (vehicles / 15 min)")
    ax.set_title("Best MAE per Model (optimal window shown)")
    ax.set_ylim(0, max(best_rows[m]["MAE_veh"] for m in models) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "best_mae_per_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# ═══════════════════════════════════════════════════════════════════════════
# Console summary table
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary(results: list[dict]) -> None:
    models = list(MODEL_REGISTRY)
    windows = sorted(set(r["window_size"] for r in results))

    lookup = {(r["model"], r["window_size"]): r for r in results}

    print(f"\n{'═' * 75}")
    print("  WINDOW-SIZE EXPERIMENT — SUMMARY (Test MAE normalised)")
    print(f"{'═' * 75}")
    header = f"{'Window':>8}  {'LB(min)':>8}" + \
             "".join(f"  {m.upper():>13}" for m in models)
    print(header)
    print("─" * 75)
    for w in windows:
        row = f"{w:>8}  {w * 15:>8}"
        for m in models:
            r = lookup.get((m, w))
            row += f"  {r['MAE']:>7.4f} ({r['MAE_veh']:>5.1f})" if r else f"  {'n/a':>13}"
        print(row)
    print("─" * 75)
    print("  Values shown as MAE_norm (MAE_vehicles)\n")

    # Best window per model
    print("  Best window per model:")
    for m in models:
        best = min((r for r in results if r["model"] == m), key=lambda x: x["MAE"])
        print(f"    {m.upper():12s} → W={best['window_size']:2d} "
              f"({best['window_size'] * 15} min)  "
              f"MAE={best['MAE']:.4f}  MAE_veh={best['MAE_veh']:.1f}")
    print(f"{'═' * 75}\n")

# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Window-size sensitivity experiment for traffic flow models"
    )
    parser.add_argument(
        "--window-sizes", type=int, nargs="+", default=WINDOW_SIZES,
        metavar="W",
        help="List of window sizes to evaluate (default: 4 8 12 16 24 32 48)",
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAIN_CFG["epochs"],
        help="Max epochs per training run (default: 50)",
    )
    parser.add_argument(
        "--patience", type=int, default=TRAIN_CFG["patience"],
        help="Early-stopping patience (default: 7)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plot generation",
    )
    args = parser.parse_args()

    cfg = {**TRAIN_CFG, "epochs": args.epochs, "patience": args.patience}

    run_experiment(
        window_sizes=sorted(set(args.window_sizes)),
        cfg=cfg,
        make_plots=not args.no_plots,
    )

if __name__ == "__main__":
    main()
