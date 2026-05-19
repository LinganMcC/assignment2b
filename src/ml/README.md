# ML Models & Training — Module README

## Overview

This module implements three traffic-flow forecasting models trained on the
Boroondara SCATS October 2006 dataset.  The goal is to predict intersection
vehicle counts per 15-minute interval, which feed into travel-time estimation
between graph nodes.

---

## File structure

```
src/ml/
├── dataset.py      — data loading, feature engineering, normalisation, windowing
├── models.py       — LSTM, GRU, Transformer model definitions (PyTorch)
├── train.py        — training loop, early stopping, checkpointing
├── evaluate.py     — test-set evaluation, comparison table, diagnostic plots
└── predict.py      — inference API  predict(site_id, datetime) → flow
```

Artefacts written to `data/processed/ml_artefacts/`:
```
scaler.pkl              — fitted SiteScaler (serialised with pickle)
lstm_best.pt            — best LSTM checkpoint
gru_best.pt             — best GRU checkpoint
transformer_best.pt     — best Transformer checkpoint
{model}_history.json    — per-epoch loss/MAE curves
results.json            — final test-set metrics for all models
plots/
    training_curves.png
    prediction_sample.png
    model_comparison.png
```

---

## Quickstart

```bash
# 1. Run data pipeline (Person A)
python -m src.data.parse_scats

# 2. Train all three models
python -m src.ml.train --all

# 3. Evaluate + generate plots
python -m src.ml.evaluate

# 4. Single prediction
python -m src.ml.predict --model gru --site-id 970 --datetime "2006-10-28 08:30:00"

# 5. Full day prediction
python -m src.ml.predict --model gru --site-id 970 --datetime "2006-10-28" --full-day
```

---

## Key design decisions

### Global vs per-site model

**Decision: global model** (one model trained on all sites simultaneously).

Rationale:
- ~75 sites × 31 days × 96 intervals ≈ 220 k rows total
- Per-site gives only ~2 900 rows each → after val/test split, only ~2 000
  training rows per site, which is insufficient for reliable recurrent training
- A global model generalises to unseen sites and is easier to maintain
- Site identity is implicitly encoded via site-specific flow distributions
  (after per-site MinMax normalisation)

Trade-off: a per-site model *would* likely achieve lower MAE for well-observed
sites (no cross-site noise), but at the cost of 75× the training runs,
75× the saved weights, and fragility to missing data.

### Input features

Per timestep (window_size = 12 steps = 3 hours look-back):

| Feature       | Description                              | Why                                   |
|---------------|------------------------------------------|---------------------------------------|
| flow_norm     | MinMax-scaled vehicle count              | Core signal                           |
| hour_sin/cos  | Cyclic hour-of-day encoding              | Avoid 23:45 → 00:00 discontinuity     |
| dow_sin/cos   | Cyclic day-of-week encoding              | Weekly seasonality                    |
| is_weekend    | Binary flag (Sat/Sun = 1)                | Weekend traffic pattern shift         |

### Normalisation

Per-site MinMax fit on **training data only** (no leakage):  
`flow_norm = (flow - site_min) / (site_max - site_min)`

The SiteScaler is serialised to `scaler.pkl` and used at inference time
to de-normalise predictions back to vehicle counts.

### Train/val/test split

Chronological (no shuffling across time boundary):
- Train: days 1–22  (~70 %)
- Val:   days 23–27 (~15 %) — used for early stopping & LR scheduling
- Test:  days 28–31 (~15 %) — held out until final evaluation

### Models

#### 1. LSTM
Two-layer stacked LSTM, hidden=64, dropout=0.2.  Classic baseline.
Captures long-range dependencies via cell state.

#### 2. GRU
Two-layer stacked GRU, hidden=64, dropout=0.2.  ~25% fewer parameters
than LSTM (no separate cell state), typically matches LSTM performance
on shorter sequences while training faster.

#### 3. Transformer *(third model)*
Lightweight Transformer encoder: input projection → positional encoding
→ 2 × TransformerEncoderLayer (4 heads, FF dim=128, Pre-LN) → global
average pooling → MLP head.

**Justification for including a Transformer:**
- Parallelises over sequence length at both training and inference time
  (no recurrent dependency chain) → lower latency for real-time use
- Self-attention captures arbitrary-lag dependencies directly (e.g.
  attending to the same interval 4 hours ago regardless of window size)
- Attention weights are interpretable: visualise which past intervals
  the model relies on for each prediction
- On standard traffic benchmarks (METR-LA, PeMS-BAY) Transformer
  variants match or exceed LSTM/GRU with comparable parameter budgets
- Provides a strong contrast point for the write-up: recurrent models
  vs attention-based; you can discuss inductive biases explicitly

**Contrast vs recurrent models:**
- GRU/LSTM accumulate information sequentially and may smooth over sharp
  peaks (rush-hour spikes) — the Transformer can attend directly to
  the peak-hour timestep regardless of its position in the window
- Transformer requires more data to generalise; GRU may outperform it
  on a 31-day dataset — this is an interesting empirical question to
  discuss in the report

### Training setup

- Optimiser: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: MSE (sensitive to large errors — appropriate for flow spikes)
- Evaluation metric: MAE (more interpretable; reported in vehicle counts)
- LR schedule: ReduceLROnPlateau (factor=0.5, patience=3 epochs)
- Early stopping: patience=7 epochs on val MAE
- Gradient clipping: max_norm=1.0

---

## Inference API

```python
from src.ml.predict import predict

# Spec-compliant function: predict(site_id, datetime) -> int
flow = predict(site_id=970, datetime_str="2006-10-28 08:30:00")
# Returns: 312  (estimated vehicles for that 15-min interval)

# Full predictor object for more control:
from src.ml.predict import FlowPredictor
p = FlowPredictor(model_name="transformer")
p.predict(970, "2006-10-28 08:30:00")          # single prediction
p.predict_batch(970, ["2006-10-28 08:30:00", "2006-10-28 08:45:00"])
p.predict_day(970, "2006-10-28")               # all 96 intervals as DataFrame
```

---

## Expected performance (indicative)

Based on published results on comparable datasets (Melbourne/Boroondara SCATS):

| Model       | MAE (veh/15min) | Notes                            |
|-------------|-----------------|----------------------------------|
| LSTM        | ~15–25          | Good baseline                    |
| GRU         | ~14–23          | Slightly faster, similar quality |
| Transformer | ~14–24          | Better on peak hours             |

Actual results will be in `data/processed/ml_artefacts/results.json` after training.
