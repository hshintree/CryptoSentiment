#!/usr/bin/env python3
"""
run_single.py  –  train once on #1train.csv, evaluate on EA & EB
"""

from pathlib import Path
import pandas as pd, yaml, torch, datetime as dt
from train_one import SingleTrainer, save_model, _coerce_dates, plot_training_history
from model          import Model

CFG       = "config.yaml"
EA_CSV    = Path("data/#1train.csv")   # 2020 tweets
EB_CSV    = Path("data/#2val.csv")     # 2015-19 & 2021-23 events

# 1 ── load data -------------------------------------------------------------
ea = pd.read_csv(EA_CSV)
eb = pd.read_csv(EB_CSV)

ea = _coerce_dates(ea)
eb = _coerce_dates(eb)


print(f"EA rows: {len(ea):,}  EB rows: {len(eb):,}")

# 2 ── model + trainer -------------------------------------------------------
cfg  = yaml.safe_load(open(CFG))
mdl  = Model(cfg["model"])

# ── pick best device automatically ─────────────────────────────────
best_device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥️  Using device: {best_device}")

trainer = SingleTrainer(mdl, CFG, device=best_device, quiet=False)
trainer.lr = 2e-5                          # then patch (works)
trainer.epochs = 3
trainer.warmup_frac = 0.20
# patch the already-built optimiser
for g in trainer.optimizer.param_groups:
    g['lr'] = trainer.lr

print(f"LR: {trainer.optimizer.param_groups[0]['lr']}")
print(f"Epochs & warmup: {trainer.epochs} at {trainer.warmup_frac}")
# 3 ── train -----------------------------------------------------------------
print("\n🚂 training on EA …")
trainer.fit(ea)

# 4 ── evaluations -----------------------------------------------------------
res_ea = trainer.evaluate(ea, name="EA (in-sample)")
res_eb = trainer.evaluate(eb, name="EB (out-of-sample)")

# 3.5 ── plot training progress ----------------------------------------------
print("\n📊 plotting training history …")
plot_training_history(trainer, outdir=Path("models")/f"single_big_model")

# 5 ── save artefacts --------------------------------------------------------
stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
save_model(mdl, Path("models")/f"single_{stamp}")

with open(f"metrics_single_{stamp}.json","w") as f:
    import json, pprint; json.dump([res_ea,res_eb], f, indent=2)
print("✓ metrics written")

