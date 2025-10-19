# recogniserai/core/energy.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class EnergyCoeffs:
    # simple triangular profile by default
    up: float = 0.02
    dn: float = 0.015
    jitter: float = 0.0  # keep 0.0 unless you want synthetic noise
    emax: float = 1.0

@dataclass
class EnergyTracker:
    """
    Simulated energy with noise-suppression:
      - EMA smoothing on the *state*
      - bounded per-epoch rate of change
    """
    coeffs: EnergyCoeffs = field(default_factory=EnergyCoeffs)
    ema: float = 0.6
    max_rate_up: float = 0.15    # max fractional increase per step
    max_rate_dn: float = 0.15    # max fractional decrease per step
    _e: float = 0.5

    def value(self) -> float:
        return float(self._e)
    
    def normalized(self):
        if not self.history:
            return 0.0
        recent = np.array(self.history[-20:])
        base = np.mean(recent)
        std = np.std(recent) + 1e-6
        return float((recent[-1] - base) / std)

    def update(self, lr: float) -> float:
        # “consume” energy when lr is high, “recover” when lr is tiny
        want = np.clip(1.0 - 4.0 * lr, 0.0, 1.0)  # heuristic demand signal
        # target energy moves towards 1.0 when want ~1, else towards 0.0
        target = want

        # smooth target step
        e_prop = (1.0 - self.ema) * self._e + self.ema * target

        # rate bounding
        max_up = self.max_rate_up * max(1e-6, self._e)
        max_dn = self.max_rate_dn * max(1e-6, self._e)

        delta = e_prop - self._e
        delta = float(np.clip(delta, -max_dn, max_up))

        # optional tiny jitter
        if self.coeffs.jitter > 0:
            delta += np.random.normal(scale=self.coeffs.jitter)

        self._e = float(np.clip(self._e + delta, 0.0, 1.0))
        return self._e  # return current energy