# recogniserai/core/efficiency.py
from __future__ import annotations
from collections import deque
import numpy as np

class EfficiencyMeter:
    """
    Tracks efficiency η_R from (loss drop) / (energy delta).
    Provides:
      - eta_smooth: EMA-smoothed efficiency
      - eta_norm:   normalised to rolling baseline (~1.0 around recent mean)
    """
    def __init__(self, smoothing: int = 5, ema: float = 0.25):
        self.smoothing = max(1, int(smoothing))
        self.ema = float(np.clip(ema, 0.01, 0.9))
        self.loss_prev = None
        self.prev_loss = None
        self.eta_ema = 0.0
        self.log = deque(maxlen=128)

        self.loss_hist = []
        self.energy_hist = []
        self.eta_hist = []
        self.eta_smooth = 0.0
        self.eta_norm = 0.0
        self.eff_log = []

    def update(self, loss_value: float, energy_delta: float) -> tuple[float, float]:
        """Update the adaptive efficiency based on loss change and energy delta."""

        loss_value = float(loss_value)
        energy_delta = float(energy_delta)

        # --- First call: initialize state ---
        if not hasattr(self, "prev_loss") or self.prev_loss is None:
            self.prev_loss = loss_value
            self.loss_hist = [loss_value]
            self.energy_hist = [max(energy_delta, 0.0)]
            self.eff_hist = [0.0]
            return 0.0, 0.0

        # --- Compute improvement ---
        d_loss = self.prev_loss - loss_value  # positive = improvement
        d_loss_pos = max(0.0, d_loss)

        # --- Safe denominator ---
        denom = max(energy_delta, 1e-8)
        eta_raw = d_loss_pos / denom if denom > 0 else 0.0
        eta_raw = float(np.clip(eta_raw, 0.0, 10.0))

        # --- Smooth via EMA ---
        self.prev_loss = loss_value
        self.loss_hist.append(loss_value)
        self.energy_hist.append(max(energy_delta, 0.0))

        self.eta_ema = (1.0 - self.ema) * self.eta_ema + self.ema * eta_raw
        self.eff_log.append(self.eta_ema)

        # --- Normalize relative to recent baseline ---
        norm_window = 50
        base = np.mean(self.eff_log[-norm_window:]) if len(self.eff_log) >= norm_window else 1.0
        eta_norm = self.eta_ema / (base + 1e-8)

        return float(eta_norm), float(self.eta_ema)