"""
dataset.py  –  Build windowed sequences from flow_timeseries.csv for model training.

Schema expected from parse_scats.py
    site_id  | datetime            | flow
    int      | YYYY-MM-DD HH:MM:SS | int

- **Input features per timestep** (window_size steps, default 12 = 3 hours):
      flow_norm   – MinMax-scaled flow for the site
      hour_sin/cos  – cyclic encoding of hour-of-day
      dow_sin/cos   – cyclic encoding of day-of-week
      is_weekend    – binary flag

      train  : days  1-22
      val    : days 23-27
      test   : days 28-31
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import pickle

# ── project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMESERIES_CSV = PROJECT_ROOT / "data" / "processed" / "flow_timeseries.csv"
ARTEFACTS_DIR = PROJECT_ROOT / "data" / "processed" / "ml_artefacts"

# ── hyper-parameters (importable constants) ──────────────────────────────────
WINDOW_SIZE = 12      # look-back steps  (12 × 15 min = 3 h)
HORIZON     = 1       # steps ahead to predict
TRAIN_END   = 22      # last day (of month) included in train
VAL_END     = 27      # last day included in val  (28-31 → test)



# Feature engineering


def _cyclic(val: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a periodic variable as (sin, cos) to avoid discontinuity."""
    angle = 2 * np.pi * val / period
    return np.sin(angle), np.cos(angle)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/dow cyclic + weekend flag columns in-place. Returns df."""
    dt = pd.to_datetime(df["datetime"])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    dow  = dt.dt.dayofweek          # 0=Mon 6=Sun

    df["hour_sin"], df["hour_cos"] = _cyclic(hour, 24.0)
    df["dow_sin"],  df["dow_cos"]  = _cyclic(dow, 7.0)
    df["is_weekend"] = (dow >= 5).astype(np.float32)
    return df



# Normalisation

class SiteScaler:
    """Per-site MinMax scaler for the 'flow' column.

    Stores {site_id: (min_val, max_val)} so we can invert at inference time.
    Falls back to global stats for unseen sites (good-enough for demo).
    """

    def __init__(self) -> None:
        self._params: Dict[int, Tuple[float, float]] = {}
        self._global: Tuple[float, float] = (0.0, 1.0)

    def fit(self, df: pd.DataFrame) -> "SiteScaler":
        """Fit on a slice of the dataframe (typically train rows only)."""
        stats = df.groupby("site_id")["flow"].agg(["min", "max"])
        for sid, row in stats.iterrows():
            lo, hi = float(row["min"]), float(row["max"])
            self._params[int(sid)] = (lo, max(hi - lo, 1.0))   # avoid /0
        # global fallback
        lo_g = float(df["flow"].min())
        hi_g = float(df["flow"].max())
        self._global = (lo_g, max(hi_g - lo_g, 1.0))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'flow_norm' column (in-place copy). Returns df."""
        df = df.copy()
        normed = np.empty(len(df), dtype=np.float32)
        for i, (sid, flow) in enumerate(zip(df["site_id"], df["flow"])):
            lo, rng = self._params.get(int(sid), self._global)
            normed[i] = (flow - lo) / rng
        df["flow_norm"] = normed
        return df

    def inverse_transform(self, site_id: int, flow_norm: np.ndarray) -> np.ndarray:
        """Convert normalised predictions back to raw vehicle counts."""
        lo, rng = self._params.get(int(site_id), self._global)
        return flow_norm * rng + lo

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "SiteScaler":
        with open(path, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Windowing
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_COLS = ["flow_norm", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
N_FEATURES   = len(FEATURE_COLS)   # 6


def _windows_for_site(
    site_df: pd.DataFrame,
    window_size: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) windows from one site's chronologically sorted dataframe.

    X shape: (n_windows, window_size, N_FEATURES)
    y shape: (n_windows,)           — scalar flow_norm target
    """
    feat = site_df[FEATURE_COLS].values.astype(np.float32)   # (T, F)
    target = site_df["flow_norm"].values.astype(np.float32)

    n = len(feat) - window_size - horizon + 1
    if n <= 0:
        return np.empty((0, window_size, N_FEATURES), dtype=np.float32), np.empty(0, dtype=np.float32)

    X = np.lib.stride_tricks.sliding_window_view(feat, (window_size, N_FEATURES))
    # shape after sliding_window_view: (n, 1, window_size, N_FEATURES) → squeeze axis 1
    X = X[:n, 0, :, :]

    y = target[window_size + horizon - 1: window_size + horizon - 1 + n]
    return X, y


def build_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build windows across all sites, returning (X, y, site_ids).

    site_ids lets us trace predictions back to specific intersections.
    """
    X_list, y_list, s_list = [], [], []
    for sid, grp in df.groupby("site_id", sort=True):
        grp = grp.sort_values("datetime")
        X_s, y_s = _windows_for_site(grp, window_size, horizon)
        if len(X_s):
            X_list.append(X_s)
            y_list.append(y_s)
            s_list.append(np.full(len(X_s), int(sid), dtype=np.int32))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    s = np.concatenate(s_list, axis=0)
    return X, y, s


# ═══════════════════════════════════════════════════════════════════════════
# Public API: load_datasets()
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(
    csv_path: Path = TIMESERIES_CSV,
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON,
    save_scaler: bool = True,
) -> dict:
    """Full pipeline: read CSV → features → normalise → window → split.

    Returns a dict with keys:
        X_train, y_train, sites_train
        X_val,   y_val,   sites_val
        X_test,  y_test,  sites_test
        scaler   (SiteScaler, fit on train only)
        n_features  (int)
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}   Sites: {df['site_id'].nunique()}")

    # ── time features ────────────────────────────────────────────────────
    df = add_time_features(df)
    df["_day"] = pd.to_datetime(df["datetime"]).dt.day

    # ── chronological split (day boundaries, no leakage) ─────────────────
    train_df = df[df["_day"] <= TRAIN_END]
    val_df   = df[(df["_day"] > TRAIN_END) & (df["_day"] <= VAL_END)]
    test_df  = df[df["_day"] > VAL_END]

    print(f"  Train days 1-{TRAIN_END}:  {len(train_df):,} rows")
    print(f"  Val   days {TRAIN_END+1}-{VAL_END}: {len(val_df):,} rows")
    print(f"  Test  days {VAL_END+1}-31: {len(test_df):,} rows")

    # ── normalise (fit only on train) ────────────────────────────────────
    scaler = SiteScaler().fit(train_df)
    train_df = scaler.transform(train_df)
    val_df   = scaler.transform(val_df)
    test_df  = scaler.transform(test_df)

    if save_scaler:
        scaler_path = ARTEFACTS_DIR / "scaler.pkl"
        scaler.save(scaler_path)
        print(f"  Scaler saved → {scaler_path}")

    # ── build windows ────────────────────────────────────────────────────
    print("Building sliding windows ...")
    X_tr, y_tr, s_tr = build_windows(train_df, window_size, horizon)
    X_va, y_va, s_va = build_windows(val_df,   window_size, horizon)
    X_te, y_te, s_te = build_windows(test_df,  window_size, horizon)

    print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    return {
        "X_train": X_tr, "y_train": y_tr, "sites_train": s_tr,
        "X_val":   X_va, "y_val":   y_va, "sites_val":   s_va,
        "X_test":  X_te, "y_test":  y_te, "sites_test":  s_te,
        "scaler":  scaler,
        "n_features": N_FEATURES,
        "window_size": window_size,
    }


if __name__ == "__main__":
    ds = load_datasets()
    print("\nDataset shapes:")
    for k in ("X_train", "X_val", "X_test"):
        print(f"  {k}: {ds[k].shape}")
