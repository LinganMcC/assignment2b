"""
train.py  –  Training pipeline for all three traffic-flow models.

Usage (from project root):
    python -m src.ml.train --model lstm
    python -m src.ml.train --model gru
    python -m src.ml.train --model transformer
    python -m src.ml.train --all          # train all three sequentially

Outputs (per model, in data/processed/ml_artefacts/):
    {model}_best.pt        – best checkpoint (lowest val MAE)
    {model}_history.json   – per-epoch train/val loss & MAE

Final evaluation on the held-out test set is printed and also saved to
    data/processed/ml_artefacts/results.json
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

from dataset import load_datasets, ARTEFACTS_DIR
from models import build_model, MODEL_REGISTRY

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── default hyper-parameters (override via CLI) ──────────────────────────────
DEFAULTS = {
    "batch_size":    256,
    "epochs":        50,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "patience":      7,       # early stopping patience (val MAE)
    "clip_grad":     1.0,     # gradient clipping max norm
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).unsqueeze(1),   # (N,) → (N,1) to match model output
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# Train one model
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    model_name: str,
    datasets: dict,
    cfg: dict,
    artefacts_dir: Path = ARTEFACTS_DIR,
) -> dict:
    """Train a single model.  Returns history dict."""

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}   device={DEVICE}")
    print(f"{'='*60}")

    artefacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artefacts_dir / f"{model_name}_best.pt"

    # ── data loaders ──────────────────────────────────────────────────────
    train_loader = _to_loader(datasets["X_train"], datasets["y_train"],
                              cfg["batch_size"], shuffle=True)
    val_loader   = _to_loader(datasets["X_val"],   datasets["y_val"],
                              cfg["batch_size"], shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────
    model = build_model(model_name, n_features=datasets["n_features"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── optimiser + schedule ──────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # ReduceLROnPlateau: halve LR if val loss doesn't improve for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    # ── training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # — train —
        model.train()
        tr_loss, tr_mae, n_batches = 0.0, 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            optimiser.step()
            tr_loss += loss.item()
            tr_mae  += _mae(pred.detach(), y_b)
            n_batches += 1
        tr_loss /= n_batches
        tr_mae  /= n_batches

        # — validate —
        model.eval()
        va_loss, va_mae, n_vbatches = 0.0, 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(X_b)
                va_loss += criterion(pred, y_b).item()
                va_mae  += _mae(pred, y_b)
                n_vbatches += 1
        va_loss /= n_vbatches
        va_mae  /= n_vbatches

        scheduler.step(va_loss)

        elapsed = time.time() - t0
        lr_now  = optimiser.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{cfg['epochs']}  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"val_MAE={va_mae:.4f}  lr={lr_now:.2e}  [{elapsed:.1f}s]")

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(va_loss, 6))
        history["train_mae"].append(round(tr_mae, 6))
        history["val_mae"].append(round(va_mae, 6))

        # — checkpoint & early stopping —
        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
            print(f"    ✓ New best val MAE={best_val_mae:.4f}  → saved {ckpt_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping after {epoch} epochs (patience={cfg['patience']})")
                break

    # ── save history ──────────────────────────────────────────────────────
    hist_path = artefacts_dir / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved → {hist_path.name}")

    return history


# ═══════════════════════════════════════════════════════════════════════════
# Evaluate on test set
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model_name: str,
    datasets: dict,
    cfg: dict,
    artefacts_dir: Path = ARTEFACTS_DIR,
) -> dict[str, float]:
    """Load best checkpoint and evaluate on the test split.

    Returns dict with MSE, MAE, RMSE (all in normalised space)
    and also MAE_vehicles (de-normalised, averaged across sites).
    """
    ckpt_path = artefacts_dir / f"{model_name}_best.pt"
    model = build_model(model_name, n_features=datasets["n_features"]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    loader = _to_loader(datasets["X_test"], datasets["y_test"],
                        cfg["batch_size"], shuffle=False)

    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds_list.append(model(X_b.to(DEVICE)).cpu().numpy())
            targets_list.append(y_b.numpy())

    preds   = np.concatenate(preds_list).squeeze()
    targets = np.concatenate(targets_list).squeeze()

    mse  = float(np.mean((preds - targets) ** 2))
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))

    # De-normalise using scaler to report in vehicle counts
    scaler = datasets["scaler"]
    site_ids = datasets["sites_test"]
    denorm_preds   = np.array([scaler.inverse_transform(int(s), np.array([p]))[0]
                                for s, p in zip(site_ids, preds)])
    denorm_targets = np.array([scaler.inverse_transform(int(s), np.array([t]))[0]
                                for s, t in zip(site_ids, targets)])
    mae_vehicles = float(np.mean(np.abs(denorm_preds - denorm_targets)))

    metrics = {
        "model":        model_name,
        "test_MSE":     round(mse,  6),
        "test_MAE":     round(mae,  6),
        "test_RMSE":    round(rmse, 6),
        "test_MAE_veh": round(mae_vehicles, 2),
    }
    print(f"\n  [{model_name.upper()}] Test metrics:")
    for k, v in metrics.items():
        if k != "model":
            print(f"    {k}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train traffic flow models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(MODEL_REGISTRY),
                       help="Train a single model")
    group.add_argument("--all", action="store_true",
                       help="Train all models sequentially")
    parser.add_argument("--epochs",    type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size",type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULTS["lr"])
    parser.add_argument("--patience",  type=int,   default=DEFAULTS["patience"])
    args = parser.parse_args()

    cfg = {
        "batch_size":   args.batch_size,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "weight_decay": DEFAULTS["weight_decay"],
        "patience":     args.patience,
        "clip_grad":    DEFAULTS["clip_grad"],
    }

    # Load data once; share across models
    datasets = load_datasets()

    models_to_train = list(MODEL_REGISTRY) if args.all else [args.model]
    all_results = []

    for name in models_to_train:
        train_model(name, datasets, cfg)
        metrics = evaluate_model(name, datasets, cfg)
        all_results.append(metrics)

    # Save combined results
    results_path = ARTEFACTS_DIR / "results.json"
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved → {results_path}")

    # Print comparison table
    if len(all_results) > 1:
        print("\n{'─'*55}")
        print(f"{'Model':<14} {'MAE (norm)':>12} {'RMSE (norm)':>12} {'MAE (veh)':>11}")
        print("─" * 52)
        for r in all_results:
            print(f"{r['model']:<14} {r['test_MAE']:>12.4f} {r['test_RMSE']:>12.4f} "
                  f"{r['test_MAE_veh']:>10.1f}")
        print("─" * 52)


if __name__ == "__main__":
    main()