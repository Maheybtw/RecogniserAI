# examples/demo_cifar.py
# Phase 3.6 – Real-data integration on CIFAR-10

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

# --- tame the noisy NVML warning (safe to keep) ---
import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch._C")
warnings.filterwarnings("ignore", message=".*pynvml.*")

import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch



# --- our adaptive bits ---
from recogniserai.core.efficiency import EfficiencyMeter
from recogniserai.core.energy import EnergyTracker, EnergyCoeffs
from recogniserai.core.controller import AdaptiveController

import torch
import os

import torch
import os

import torch, sys, os

# --- Device setup utility (final release version) ---
import torch, sys
import multiprocessing

def get_device():
    """Return CUDA device if available, else CPU — prints once per main process."""
    if hasattr(sys, "_DEVICE"):
        return sys._DEVICE  # Don't redo detection or printing

    is_main = multiprocessing.current_process().name == "MainProcess"

    if torch.cuda.is_available():
        sys._DEVICE = torch.device("cuda")
        if is_main:
            print(f"\n✅ Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        sys._DEVICE = torch.device("cpu")
        if is_main:
            print("\n⚠ CUDA not available, using CPU instead.\n")

    return sys._DEVICE

DEVICE = get_device()








OUT_DIR = os.path.join(os.path.dirname(__file__), "phase3_results")
os.makedirs(OUT_DIR, exist_ok=True)



# ----------------------------
# Utilities
# ----------------------------
@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return 100.0 * correct / max(1, total)

def run_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total_loss / max(1, n)

# ----------------------------
# Result container
# ----------------------------
@dataclass
class RunResult:
    name: str
    eta:  List[float]
    lr:   List[float]
    acc:  List[float]

# ----------------------------
# Baseline: Fixed LR
# ----------------------------
def train_fixed(model_fn, train_loader, val_loader, epochs=50, lr=1e-3) -> RunResult:
    name = "FixedLR_1e-3"
    model = model_fn().to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    eff    = EfficiencyMeter(smoothing=5, ema=0.25)
    energy = EnergyTracker(coeffs=EnergyCoeffs(up=0.012, dn=0.012, jitter=0.002, emax=1.0))

    


    eta_curve, lr_curve, acc_curve = [], [], []
    for ep in range(epochs):
        mean_loss = run_epoch(model, train_loader, opt, loss_fn)
        _ = energy.update(float(opt.param_groups[0]["lr"]))         # keep energy in play
        _, eta_smooth = eff.update(float(mean_loss), _)
        val_acc = accuracy(model, val_loader)

        eta_curve.append(float(eta_smooth))
        lr_curve.append(float(opt.param_groups[0]["lr"]))
        acc_curve.append(val_acc)

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[{name}] Epoch {ep:03d} | η_R: {eta_smooth:.4f} | LR: {opt.param_groups[0]['lr']:.5f} | Acc: {val_acc:.2f}%")

    return RunResult(name, eta_curve, lr_curve, acc_curve)

# ----------------------------
# Baseline: Cosine Anneal
# ----------------------------
def train_cosine(model_fn, train_loader, val_loader, epochs=50, lr_max=1e-1, lr_min=1e-5) -> RunResult:
    name = "Cosine_0.1_1e-5"
    model = model_fn().to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=lr_max, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)
    loss_fn = nn.CrossEntropyLoss()

    eff    = EfficiencyMeter(smoothing=5, ema=0.25)
    energy = EnergyTracker(coeffs=EnergyCoeffs(up=0.012, dn=0.012, jitter=0.002, emax=1.0))

    


    eta_curve, lr_curve, acc_curve = [], [], []
    for ep in range(epochs):
        mean_loss = run_epoch(model, train_loader, opt, loss_fn)
        _ = energy.update(float(opt.param_groups[0]["lr"]))
        _, eta_smooth = eff.update(float(mean_loss), _)
        sched.step()
        val_acc = accuracy(model, val_loader)

        eta_curve.append(float(eta_smooth))
        lr_curve.append(float(opt.param_groups[0]["lr"]))
        acc_curve.append(val_acc)

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[{name}] Epoch {ep:03d} | η_R: {eta_smooth:.4f} | LR: {opt.param_groups[0]['lr']:.5f} | Acc: {val_acc:.2f}%")

    return RunResult(name, eta_curve, lr_curve, acc_curve)

# ----------------------------
# Baseline: OneCycle
# ----------------------------
def train_onecycle(model_fn, train_loader, val_loader, epochs=50, max_lr=1e-1, base_lr=1e-5) -> RunResult:
    name = "OneCycle_0.1_1e-5"
    model = model_fn().to(DEVICE)
    opt   = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=epochs, pct_start=0.3, anneal_strategy="cos")
    loss_fn = nn.CrossEntropyLoss()

    eff    = EfficiencyMeter(smoothing=5, ema=0.25)
    energy = EnergyTracker(coeffs=EnergyCoeffs(up=0.012, dn=0.012, jitter=0.002, emax=1.0))
    
    


    eta_curve, lr_curve, acc_curve = [], [], []
    for ep in range(epochs):
        mean_loss = run_epoch(model, train_loader, opt, loss_fn)
        _ = energy.update(float(opt.param_groups[0]["lr"]))
        _, eta_smooth = eff.update(float(mean_loss), _)
        sched.step()
        val_acc = accuracy(model, val_loader)

        eta_curve.append(float(eta_smooth))
        lr_curve.append(float(opt.param_groups[0]["lr"]))
        acc_curve.append(val_acc)

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[{name}] Epoch {ep:03d} | η_R: {eta_smooth:.4f} | LR: {opt.param_groups[0]['lr']:.5f} | Acc: {val_acc:.2f}%")

    return RunResult(name, eta_curve, lr_curve, acc_curve)

# ----------------------------
# Adaptive
# ----------------------------
def train_adaptive(model_fn, train_loader, val_loader, epochs=50) -> RunResult:
    name = "Adaptive"
    model = model_fn().to(DEVICE)

    # start modest; controller will scale safely
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    eff    = EfficiencyMeter(smoothing=5, ema=0.25)
    energy = EnergyTracker(coeffs=EnergyCoeffs(up=0.012, dn=0.012, jitter=0.002, emax=1.0))

    


    # --- tuned for CIFAR-10 (stable & brisk growth) ---
    ctrl = AdaptiveController(
        target_eff=0.72,
        lr_bounds=(1e-6, 2e-2),
        # core gains
        k_p=0.35, k_i=0.05, ema=0.20, leak=0.01,
        # memory / events
        mem_gate=0.65, mem_gain=0.12,
        event_gain=0.18, event_cap=0.30,
        # floor recovery
        floor_patience=24, floor_kick=3.0, cool_down=6,
        # bootstrap + dynamic kp (your current controller supports these)
        bootstrap_steps=600, bootstrap_gain=0.10,
        kp_min=0.05, kp_max=0.50, kp_alpha=0.20,
        # derivative damping (if present in your version)
        d_gain=0.30, _d_ema=0.35
    )
    # --- feedback log (for visualization after training) ---
    ctrl.log = {"epoch": [], "eta_smooth": [], "lr": [], "eff_err": [], "val_acc": []}

    ctrl.reset()
    print(f"Initial LR before training: {opt.param_groups[0]['lr']:.6f}")

    eta_curve, lr_curve, acc_curve = [], [], []
    mem_state_prev = 0.0

    for ep in range(epochs):
        mean_loss = run_epoch(model, train_loader, opt, loss_fn)

        # --- energy + η_R (efficiency) update ---
        e_val = energy.update(float(opt.param_groups[0]["lr"]))
        _, eta_smooth = eff.update(float(mean_loss), e_val)

        # --- event/memory signals (small, real, helpful) ---
        val_acc = accuracy(model, val_loader)
        # a tiny salience when accuracy jumps, plus slow memory trace
        acc_jump = 0.01 * max(0.0, (val_acc - acc_curve[-1]) if acc_curve else 0.0)
        mem_state = 0.98 * mem_state_prev + 0.02 * (val_acc / 100.0)
        mem_state_prev = mem_state

        new_lr = ctrl.update(
            opt,
            eta_smooth,
            memory_signal=float(mem_state),
            salience=float(acc_jump)
        )
        # safety clip (controller already clips, but keep parity with MNIST demo)
        opt.param_groups[0]["lr"] = float(np.clip(new_lr, ctrl.lr_bounds[0], ctrl.lr_bounds[1]))

            # --- log controller state each epoch ---
        ctrl.log["epoch"].append(ep)
        ctrl.log["eta_smooth"].append(float(eta_smooth))
        ctrl.log["lr"].append(float(opt.param_groups[0]['lr']))
        ctrl.log["eff_err"].append(float(getattr(ctrl, "err", 0.0)))
        ctrl.log["val_acc"].append(float(val_acc))


        eta_curve.append(float(eta_smooth))
        lr_curve.append(float(opt.param_groups[0]["lr"]))
        acc_curve.append(val_acc)

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[{name}] Epoch {ep:03d} | η_R: {eta_smooth:.4f} | LR: {opt.param_groups[0]['lr']:.5f} | Acc: {val_acc:.2f}%")

    # --- visualize feedback loop ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(ctrl.log["epoch"], ctrl.log["lr"], label="LR", color="tab:red")
    plt.plot(ctrl.log["epoch"], ctrl.log["eta_smooth"], label="η_R (smoothed)", color="tab:blue")
    plt.plot(ctrl.log["epoch"], ctrl.log["eff_err"], label="Efficiency Error", color="tab:green", alpha=0.6)
    plt.title("Adaptive Controller Feedback Dynamics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "adaptive_feedback.png"), dpi=150)
    plt.close()

    import pandas as pd

    pd.DataFrame(ctrl.log).to_csv(
        os.path.join(OUT_DIR, "adaptive_feedback.csv"),
        index=False
    )


    print(f"Saved feedback visualization → {os.path.join(OUT_DIR, 'adaptive_feedback.png')}")


    return RunResult(name, eta_curve, lr_curve, acc_curve)

# ----------------------------
# Model factory (small ResNet18)
# ----------------------------
def ModelCIFAR():
    from torchvision.models import resnet18
    m = resnet18(num_classes=10)
    # smaller first conv for CIFAR (3x32x32)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

# ----------------------------
# Main
# ----------------------------
def main():
    # data
    tfm_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)),
    ])
    tfm_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)),
    ])

    root = os.path.join(OUT_DIR, "data")
    train = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=tfm_train)
    test  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_test)

    bs = 128
    train_loader = DataLoader(train, batch_size=bs, persistent_workers=True, shuffle=True,  num_workers=10, pin_memory=(DEVICE=="cuda"))
    test_loader  = DataLoader(test,  batch_size=256, persistent_workers=True, shuffle=False, num_workers=10, pin_memory=(DEVICE=="cuda"))

    epochs = 50

    results: List[RunResult] = []
    results.append(train_fixed   (ModelCIFAR, train_loader, test_loader, epochs=epochs, lr=1e-3))
    results.append(train_cosine  (ModelCIFAR, train_loader, test_loader, epochs=epochs, lr_max=1e-1, lr_min=1e-5))
    results.append(train_onecycle(ModelCIFAR, train_loader, test_loader, epochs=epochs, max_lr=1e-1, base_lr=1e-5))
    results.append(train_adaptive(ModelCIFAR, train_loader, test_loader, epochs=epochs))

    # ----- Save comparison plots -----
    x = np.arange(epochs)

    # η_R
    plt.figure(figsize=(12,4))
    for r in results: plt.plot(x, r.eta, label=r.name)
    plt.title("η_R (smooth) comparison"); plt.xlabel("Epoch"); plt.ylabel("η_R (smooth)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cifar_compare_eta.png"), dpi=140)

    # LR
    plt.figure(figsize=(12,4))
    for r in results: plt.plot(x, r.lr, label=r.name)
    plt.title("LR comparison"); plt.xlabel("Epoch"); plt.ylabel("LR"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cifar_compare_lr.png"), dpi=140)

    # Accuracy
    plt.figure(figsize=(12,4))
    for r in results: plt.plot(x, r.acc, label=r.name)
    plt.title("Accuracy comparison"); plt.xlabel("Epoch"); plt.ylabel("Acc (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cifar_compare_accuracy.png"), dpi=140)

    # CSV summary (simple)
    import csv
    with open(os.path.join(OUT_DIR, "cifar_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + [f"{r.name}_acc" for r in results] + [f"{r.name}_lr" for r in results] + [f"{r.name}_eta" for r in results])
        for i in range(epochs):
            row = [i] + [r.acc[i] for r in results] + [r.lr[i] for r in results] + [r.eta[i] for r in results]
            w.writerow(row)

    print(f"Done. Plots + summary saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()