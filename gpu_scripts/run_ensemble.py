#!/usr/bin/env python3
"""
run_ensemble.py - Train ensemble of models with different seeds and evaluate
"""

from pathlib import Path
import torch, numpy as np, yaml, pandas as pd
from datetime import datetime
from gpu_scripts.train_one import SingleTrainer, save_model, _coerce_dates, predict_ensemble, load_ensemble
from gpu_scripts.model import Model

CFG       = "config.yaml"
EA        = _coerce_dates(pd.read_csv("data/#1train.csv"))
EB        = _coerce_dates(pd.read_csv("data/#2val.csv"))
SEEDS     = [42, 1337, 2025]
CKPTS     = Path("models/ensemble"); CKPTS.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(CFG))

# ── pick best device automatically ─────────────────────────────────
best_device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

# ---------- train & save -------------------------------------------------
for s in SEEDS:
    print(f"\n🕒  {datetime.now():%H:%M:%S}  — seed {s} starting")

    torch.manual_seed(s);  np.random.seed(s)
    m = Model(cfg["model"])
    t = SingleTrainer(m, CFG, device=best_device, quiet=False)
    t.lr = 2e-5; t.epochs = 3; t.warmup_frac = 0.20
    t.fit(EA)
    save_model(m, CKPTS / f"seed{s}")

    print(f"✅  {datetime.now():%H:%M:%S}  — seed {s} finished")

# ---------- inference ----------------------------------------------------
models  = load_ensemble(CKPTS, CFG)
helper = SingleTrainer(
    Model(cfg["model"]), CFG,
    device=best_device,
    quiet=False,           # only for preprocessing/loader
)

print("\n=== ENSEMBLE RESULTS ===")
predict_ensemble(models, EA, helper, name="EA-ens")
predict_ensemble(models, EB, helper, name="EB-ens") 