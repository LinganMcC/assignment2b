"""
models.py  –  Three traffic-flow forecasting models.

All models share the same interface:
    - Input  shape: (batch, window_size, n_features)
    - Output shape: (batch, 1)   — single-step normalised flow prediction

Models
------
1. LSTMModel     – stacked LSTM with dropout  (classic sequence baseline)
2. GRUModel      – stacked GRU  with dropout  (faster, fewer params, usually ≈LSTM quality)
3. TransformerModel – lightweight Transformer encoder + MLP head
                    Justified because:
                    (a) no recurrent bottleneck → parallelises well at inference
                    (b) attention weights are interpretable (which past timesteps matter?)
                    (c) literature shows competitive accuracy on short traffic horizons
                    (d) easy to contrast: "better at capturing periodic peaks (rush hour)
                        while GRU/LSTM may smooth over them"

Framework: PyTorch (chosen over Keras for flexibility; torch.compile speeds training
on modern hardware and PyTorch is now dominant in research).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# 1. LSTM
# ═══════════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """Two-layer stacked LSTM → linear output.

    Args:
        n_features:  number of input channels per timestep (default 6)
        hidden_size: LSTM hidden dimension
        num_layers:  depth of stacked LSTM
        dropout:     dropout between LSTM layers (and before output head)
    """

    def __init__(
        self,
        n_features: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last   = out[:, -1, :]         # take final timestep
        return self.head(self.dropout(last))   # (B, 1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. GRU
# ═══════════════════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    """Two-layer stacked GRU → linear output.

    Structurally identical to LSTMModel; GRU has ~25 % fewer parameters
    per layer (no separate cell state), which can reduce overfitting on
    moderate-sized datasets.
    """

    def __init__(
        self,
        n_features: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        last   = out[:, -1, :]
        return self.head(self.dropout(last))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Transformer
# ═══════════════════════════════════════════════════════════════════════════

class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerModel(nn.Module):
    """Lightweight Transformer encoder for time-series regression.

    Architecture:
        Input projection (n_features → d_model)
        → Positional encoding
        → N × TransformerEncoderLayer (multi-head self-attention + FF)
        → Global average pooling over time dimension
        → MLP head (d_model → hidden → 1)

    Justification vs LSTM/GRU:
      - Self-attention captures arbitrary-range dependencies (e.g. yesterday's
        same hour) without sequential bottleneck.
      - Attention weights interpretable: can visualise which past intervals the
        model attends to for a given prediction.
      - On benchmark traffic datasets (METR-LA, PeMS) Transformer variants match
        or exceed LSTM/GRU with similar or fewer parameters.
    """

    def __init__(
        self,
        n_features: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, F)
        z = self.pos_enc(self.input_proj(x))   # (B, T, d_model)
        z = self.encoder(z)                    # (B, T, d_model)
        z = z.mean(dim=1)                      # global avg pool → (B, d_model)
        return self.head(z)                    # (B, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, type] = {
    "lstm":        LSTMModel,
    "gru":         GRUModel,
    "transformer": TransformerModel,
}


def build_model(name: str, n_features: int = 6, **kwargs) -> nn.Module:
    """Instantiate a model by name.  Extra kwargs forwarded to constructor.

    Example:
        model = build_model("lstm", n_features=6, hidden_size=128)
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_features=n_features, **kwargs)


if __name__ == "__main__":
    # Smoke-test all three models
    B, T, F = 8, 12, 6
    x = torch.randn(B, T, F)
    for name in MODEL_REGISTRY:
        m = build_model(name, n_features=F)
        out = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name:12s}  output={tuple(out.shape)}  params={n_params:,}")
