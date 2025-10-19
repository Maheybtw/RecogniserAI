# recogniserai/core/controller.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
from collections import deque
import numpy as np
import math
import recogniserai

# Display a startup message once when the controller is first imported
if not getattr(recogniserai, "_banner_shown", False):
    print(f"RecogniserAI v{recogniserai.__version__} — Adaptive Controller Ready")
    recogniserai._banner_shown = True


@dataclass
class AdaptiveController:
    """
    Smooth PI controller for LR with:
      - anti-windup (no integral growth while saturated against bounds)
      - EMA smoothing on LR updates
      - optional memory gain (boosts LR when consolidated events are strong)
      - gentle floor-recovery if stuck at lr_min AND efficiency far below target
      - NEW (3.3): event-driven 'salience' boost and self-tuning proportional gain
    """

    target_eff: float = 1.0
    lr_bounds: Tuple[float, float] = (1e-6, 1.5e-1)

    # base controller gains
    k_p: float = 0.08
    k_i: float = 0.015

    # smoothing & gating
    ema: float = 0.20
    leak: float = 0.0
    mem_gate: float = 0.6
    mem_gain: float = 0.10

    # floor recovery
    floor_patience: int = 20
    floor_kick: float = 3.0
    cool_down: int = 0

    # rate limits per step (multiplicative up/down)
    max_rate_up: float = 1.20
    min_rate_dn: float = 0.80

    bootstrap_steps: int = 400
    bootstrap_gain: float = 0.10

    # --- NEW (3.3): dynamic proportional tuning ---
    kp_min: float = 0.05
    kp_max: float = 0.20
    kp_alpha: float = 0.10        # EMA for error stats
    _err_mean: float = 0.0        # running mean(|err|)
    _err_m2: float = 0.0          # running second moment for variance

    # --- NEW (3.3): event/salience boost ---
    event_gain: float = 0.15      # scale for salience in [0..1]
    event_cap: float = 0.30       # cap added to delta from salience

    # internal state
    i_term: float = 0.0
    cooldown_left: int = 0
    at_floor_count: int = 0

    # --- derivative damping (new) ---
    d_gain: float = 0.20            # strength of derivative (slope) damping
    d_beta: float = 0.80            # EMA for slope smoothing (0..1), higher = smoother
    d_hist: int = 64                # history window to judge collapses

    # --- auto-recovery (new) ---
    reset_on_collapse: bool = True  # if a big drop in η_R is detected, reset integral & cool down
    collapse_pct: float = 0.55      # trigger reset if η_R falls below this fraction of its recent max

    # --- internal (new) ---
    _eta_prev: float = 0.0
    _d_ema: float = 0.0
    from collections import deque
    from dataclasses import field

    eta_history: deque = field(default_factory=lambda: deque(maxlen=64))
    last_terms: dict = None

    d_gain: float = 0.25
    d_beta: float = 0.85
    reset_on_collapse: bool = True
    collapse_pct: float = 0.55

    def __post_init__(self):
        self.integral = 0.0
        self.k_p_dyn = self.k_p
        self.step_count = 0

    def _gate(self, x: float, t: float) -> float:
        """Smooth gate in [0,1] using logistic around threshold t."""
        s = 1.0 / (1.0 + math.exp(-10.0 * (x - t)))
        return float(s)

    def reset(self) -> None:
        """Reset internal state between runs (Phase 3.2 bootstrap hook)."""
        self.i_term = 0.0
        self.cooldown_left = 0
        self.at_floor_count = 0
        self._err_mean = 0.0
        self._err_m2 = 0.0
        # new 3.5 state
        self._eta_prev = 0.0
        self._d_ema = 0.0
        self.eta_history = deque(maxlen=max(32, self.d_hist))
        self.last_terms = {"p": 0.0, "i": 0.0, "m": 0.0, "d": 0.0, "delta": 0.0}

    def _tune_kp(self, err: float) -> float:
        """
        NEW: Self-tuning proportional gain based on recent error stability.
        Uses an EMA for mean(|err|) and variance to form a z-like score:
        - bigger |err| relative to recent noise -> higher k_p (faster response)
        - noisy/unstable -> lower k_p (more damping)
        """
        a = self.kp_alpha
        e = abs(float(err))

        # update mean and "variance" tracker (Welford-ish EMA)
        prev_mean = self._err_mean
        self._err_mean = (1.0 - a) * self._err_mean + a * e
        # squared EMA (not exact variance, but stable and cheap)
        self._err_m2 = (1.0 - a) * self._err_m2 + a * (e * e)

        # robust sigma estimate
        sigma2 = max(1e-10, self._err_m2 - self._err_mean * self._err_mean)
        sigma = math.sqrt(sigma2)

        # z-like score (how large is the current error w.r.t recent noise)
        z = e / (sigma + 1e-4)
        # map z to [0.75, 1.25] with soft saturation
        scale = 0.75 + 0.5 * math.tanh(0.5 * (z - 1.0))
        k_eff = np.clip(self.k_p * scale, self.kp_min, self.kp_max)
        return float(k_eff)

    def update(
        self,
        optimizer,
        eta_smooth: float,
        memory_signal: float = 0.0,
        salience: float = 0.0,
    ) -> float:
        """
        Update LR using smoothed efficiency and optional signals.
        Returns the *new* LR written into optimizer.param_groups[0]['lr'].
        """
        pg = optimizer.param_groups[0]
        lr_min, lr_max = self.lr_bounds
        lr = float(pg["lr"])

            # === Phase 3.4: Bootstrap & dynamic proportional gain ===
        self.step_count += 1

        # Bootstrap warmup: temporary LR expansion early in training
        if self.step_count < self.bootstrap_steps:
            lr *= 1.0 + self.bootstrap_gain * (1.0 - self.step_count / self.bootstrap_steps)

        # error: positive means "below target" -> need more LR
        # scale by target to keep numerics reasonable
        err = (self.target_eff - float(eta_smooth)) / max(1e-8, self.target_eff)
        err = float(np.clip(err, -5.0, 5.0))

        # dynamic proportional term
        # --- Dynamic proportional gain based on salience & memory ---
        sal_boost = min(self.event_cap, abs(salience) * self.event_gain + abs(memory_signal) * 0.1)
        target_kp = np.clip(self.k_p + sal_boost, self.kp_min, self.kp_max)
        self.k_p_dyn = (1.0 - self.kp_alpha) * self.k_p_dyn + self.kp_alpha * target_kp

        p_term = self.k_p_dyn * err

        # anti-windup: only integrate if not pushing at bounds
        pushing_up = (p_term + self.i_term) > 0.0 and math.isclose(lr, lr_max, rel_tol=0.0, abs_tol=1e-18)
        pushing_dn = (p_term + self.i_term) < 0.0 and math.isclose(lr, lr_min, rel_tol=0.0, abs_tol=1e-18)
        if not (pushing_up or pushing_dn):
            self.i_term += self.k_i * err
            if self.leak > 0.0:
                self.i_term *= (1.0 - self.leak)
            self.i_term = float(np.clip(self.i_term, -0.5, 0.5))

        # memory gain (smoothly activated)
        m_gate = self._gate(memory_signal, self.mem_gate)
        m_boost = self.mem_gain * m_gate


        # NEW: event/salience boost (small, short impulse on change)
        s = float(np.clip(salience, 0.0, 1.0))
        evt_boost = float(np.clip(self.event_gain * s, 0.0, self.event_cap))

        # raw delta-LR (dimensionless), squash with tanh for safety
        delta = p_term + self.i_term + m_boost + evt_boost
        delta = float(np.tanh(delta))

        # interpret as multiplicative factor via exp(alpha * delta)
        alpha = 0.25
        proposed = lr * math.exp(alpha * delta)

        # -----------------------------
        # derivative damping (Phase 3.5)
        # -----------------------------
        slope = float(eta_smooth) - getattr(self, "_eta_prev", 0.0)
        self._eta_prev = float(eta_smooth)
        self._d_ema = getattr(self, "_d_ema", 0.0)
        self._d_ema = self.d_beta * self._d_ema + (1.0 - self.d_beta) * slope
        d_term = - self.d_gain * self._d_ema  # opposes rapid rises

        # -----------------------------
        # combine terms and squash
        # -----------------------------
        raw = p_term + self.i_term + m_boost + d_term
        delta = float(np.tanh(raw))
        self.last_terms = {"p": p_term, "i": self.i_term, "m": m_boost, "d": d_term, "delta": delta}

        # -----------------------------
        # scale LR and clip
        # -----------------------------
        alpha = 0.25
        proposed = lr * math.exp(alpha * delta)
        scale = np.clip(proposed / max(lr, 1e-16), self.min_rate_dn, self.max_rate_up)
        new_lr = float(np.clip(lr * scale, lr_min, lr_max))

        # -----------------------------
        # collapse detection & auto-recovery
        # -----------------------------
        if not hasattr(self, "eta_history"):
            from collections import deque
            self.eta_history = deque(maxlen=64)
        self.eta_history.append(float(eta_smooth))

        if getattr(self, "reset_on_collapse", True) and len(self.eta_history) >= 12:
            hi = max(self.eta_history)
            if hi > 1e-12 and float(eta_smooth) < getattr(self, "collapse_pct", 0.55) * hi:
                # big drop → reset integral + cooldown
                self.i_term = 0.0
                self.cooldown_left = max(self.cooldown_left, getattr(self, "cool_down", 3))
                new_lr = float(np.clip(new_lr * 0.95, lr_min, lr_max))

            # Debug monitor (can remove later)
        if self.step_count % 100 == 0:
            print(f"[AdaptiveController] step={self.step_count} lr={lr:.6f} kp_dyn={self.k_p_dyn:.3f}")

        pg["lr"] = new_lr
        return new_lr