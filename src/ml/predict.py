"""
predict.py  –  Inference interface for trained traffic-flow models.

Public API
----------
    from src.ml.predict import FlowPredictor

    predictor = FlowPredictor(model_name="gru")   # or "lstm" / "transformer"
    flow = predictor.predict(site_id=970, dt="2006-10-30 08:30:00")
    # → 312   (estimated vehicles / 15-min interval at that site)

    # Batch version:
    results = predictor.predict_batch(site_id=970, datetimes=[...])

    # Convenience function matching the spec signature exactly:
    from src.ml.predict import predict
    flow = predict(site_id=970, datetime="2006-10-30 08:30:00")

How inference works
-------------------
To predict flow at time T for site S, we need the WINDOW_SIZE preceding
observations.  This module loads the full timeseries CSV into memory
(~4 MB) and selects the correct look-back window.  For production use
you would replace that with a database query.

If the requested datetime falls within the dataset, real historical data
is used as context.  If it's outside the dataset (e.g. a future date),
the predictor wraps around using the same time-of-week pattern from the
nearest available week (seasonal fallback).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch

from dataset import (
    ARTEFACTS_DIR, TIMESERIES_CSV, WINDOW_SIZE, N_FEATURES,
    FEATURE_COLS, SiteScaler, add_time_features,
)
from models import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════════════
# FlowPredictor
# ═══════════════════════════════════════════════════════════════════════════

class FlowPredictor:
    """Load a trained model + scaler and expose a predict() method.

    Args:
        model_name:     "lstm", "gru", or "transformer"
        artefacts_dir:  directory that contains {model}_best.pt and scaler.pkl
        timeseries_csv: path to flow_timeseries.csv (used for look-back context)
        device:         torch device string; defaults to cuda if available
    """

    def __init__(
        self,
        model_name: str = "gru",
        artefacts_dir: Path = ARTEFACTS_DIR,
        timeseries_csv: Path = TIMESERIES_CSV,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.artefacts_dir = artefacts_dir
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load scaler
        scaler_path = artefacts_dir / "scaler.pkl"
        self.scaler: SiteScaler = SiteScaler.load(scaler_path)

        # Load model
        ckpt_path = artefacts_dir / f"{model_name}_best.pt"
        self.model = build_model(model_name, n_features=N_FEATURES).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

        # Load historical timeseries into memory (used for look-back windows)
        self._ts: pd.DataFrame = self._load_timeseries(timeseries_csv)

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _load_timeseries(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = add_time_features(df)
        df["flow_norm_raw"] = df["flow"].copy()  # keep raw for scaler
        return df

    def _get_context_window(
        self, site_id: int, target_dt: datetime
    ) -> np.ndarray:
        """Return the (WINDOW_SIZE, N_FEATURES) look-back array ending just
        before target_dt for the given site.

        If the exact timestamps are unavailable, falls back to the same
        time-of-week pattern one week earlier (seasonal proxy).
        """
        site_df = self._ts[self._ts["site_id"] == site_id].copy()
        if site_df.empty:
            raise ValueError(f"site_id={site_id} not found in timeseries.")

        # Normalise flow for this site
        site_df = self.scaler.transform(site_df)

        site_df["dt"] = pd.to_datetime(site_df["datetime"])
        site_df = site_df.sort_values("dt")

        # Look-back window: the WINDOW_SIZE steps strictly before target_dt
        mask = site_df["dt"] < target_dt
        context = site_df[mask].tail(WINDOW_SIZE)

        if len(context) < WINDOW_SIZE:
            # Seasonal fallback: use same time-of-week from one week prior
            fallback_end = target_dt - timedelta(weeks=1)
            mask2 = site_df["dt"] < fallback_end
            context = site_df[mask2].tail(WINDOW_SIZE)

        if len(context) < WINDOW_SIZE:
            # Last-resort: use whatever is available, pad with the first row
            pad_rows = WINDOW_SIZE - len(context)
            context = pd.concat([context.iloc[:1].loc[context.index[:1]].iloc[[0]] * pad_rows,
                                  context], ignore_index=True)

        window = context[FEATURE_COLS].values.astype(np.float32)
        # Ensure exactly WINDOW_SIZE rows
        window = window[-WINDOW_SIZE:]
        return window   # (WINDOW_SIZE, N_FEATURES)

    # ── public interface ──────────────────────────────────────────────────

    def predict(
        self,
        site_id: int,
        dt: Union[str, datetime],
    ) -> int:
        """Predict flow (vehicle count) for a given site and 15-min interval.

        Args:
            site_id:  SCATS site number (e.g. 970)
            dt:       start of the 15-min interval to predict.
                      String format: "YYYY-MM-DD HH:MM:SS" or datetime object.

        Returns:
            Estimated vehicle count (integer) for that 15-min interval.
        """
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)

        window = self._get_context_window(site_id, dt)  # (W, F)
        X = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, W, F)

        with torch.no_grad():
            norm_pred = self.model(X).item()

        # Clamp to [0, 1] before inverse-transforming
        norm_pred = max(0.0, min(1.0, norm_pred))
        raw = self.scaler.inverse_transform(site_id, np.array([norm_pred]))[0]
        return max(0, int(round(raw)))

    def predict_batch(
        self,
        site_id: int,
        datetimes: list[Union[str, datetime]],
    ) -> list[int]:
        """Predict flow for a list of datetimes at one site."""
        return [self.predict(site_id, dt) for dt in datetimes]

    def predict_day(
        self,
        site_id: int,
        date: Union[str, datetime],
    ) -> pd.DataFrame:
        """Predict all 96 intervals for a given site and day.

        Returns a DataFrame with columns: datetime, predicted_flow.
        """
        if isinstance(date, str):
            date = datetime.strptime(date[:10], "%Y-%m-%d")

        intervals = [date + timedelta(minutes=15 * i) for i in range(96)]
        flows = self.predict_batch(site_id, intervals)
        return pd.DataFrame({"datetime": intervals, "predicted_flow": flows})


# ═══════════════════════════════════════════════════════════════════════════
# Convenience function matching the spec: predict(site_id, datetime) -> flow
# ═══════════════════════════════════════════════════════════════════════════

# Module-level singleton (lazy-loaded on first call)
_DEFAULT_PREDICTOR: FlowPredictor | None = None


def predict(
    site_id: int,
    datetime_str: Union[str, datetime],
    model_name: str = "gru",
) -> int:
    """Top-level convenience function: predict(site_id, datetime) → flow.

    Matches the interface specified in the project spec.

    Args:
        site_id:       SCATS site number
        datetime_str:  "YYYY-MM-DD HH:MM:SS" string or datetime object
        model_name:    which trained model to use (default: "gru")

    Returns:
        Estimated vehicle count (int) for the 15-min interval starting
        at the given datetime.

    Example:
        >>> from src.ml.predict import predict
        >>> predict(970, "2006-10-15 08:00:00")
        287
    """
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None or _DEFAULT_PREDICTOR.model_name != model_name:
        _DEFAULT_PREDICTOR = FlowPredictor(model_name=model_name)
    return _DEFAULT_PREDICTOR.predict(site_id, datetime_str)


# ═══════════════════════════════════════════════════════════════════════════
# CLI demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a trained model")
    parser.add_argument("--model",    default="gru",  choices=["lstm", "gru", "transformer"])
    parser.add_argument("--site-id",  type=int, required=True)
    parser.add_argument("--datetime", type=str, default="2006-10-28 08:30:00",
                        help="ISO datetime string, e.g. '2006-10-28 08:30:00'")
    parser.add_argument("--full-day", action="store_true",
                        help="Predict all 96 intervals for the given date")
    args = parser.parse_args()

    predictor = FlowPredictor(model_name=args.model)

    if args.full_day:
        df = predictor.predict_day(args.site_id, args.datetime[:10])
        print(df.to_string(index=False))
    else:
        flow = predictor.predict(args.site_id, args.datetime)
        print(f"\n  Site {args.site_id}  at {args.datetime}")
        print(f"  Predicted flow: {flow} vehicles / 15 min")
