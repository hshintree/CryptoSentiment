#!/usr/bin/env python3
"""
Smoke-test majority-vote aggregation.

• uses the real config.yaml — nothing is written to disk
• trains on 300 tweets, 1 epoch, batch=4
• verifies Pred_Label & Pred_Conf appear after cross-fold voting
"""

import yaml, torch
from dataset_loader import DatasetLoader
from preprocessor    import Preprocessor
from market_labeler  import MarketLabeler
from model           import Model
from trainer         import Trainer, cross_val_predict          # <- new helper

CFG = "config.yaml"

# ── 1. grab a tiny slice so the test is fast ─────────────────────────────
dl  = DatasetLoader(CFG)
pp  = Preprocessor(CFG)
ml  = MarketLabeler(CFG)

data = dl.load_dataset(aggregate=False).head(300)
data = ml.label_data(pp.preprocess(data))
data["Previous Label"] = data["Label"].shift(1).fillna("Neutral")

# ── 2. build model & trainer, then DOWNSCALE hyper-params in memory ─────
with open(CFG) as f:
    params = yaml.safe_load(f)["model"]
model = Model(params)

trainer = Trainer(model, data, CFG)
trainer.batch_size = 4          # override heavy defaults without touching YAML
trainer.epochs     = 1

print("⚡ mini-train starting (300 rows, 1 epoch, batch=4)…")
trainer.train()
print("✓ mini-train done")

# ── 3. majority-vote aggregation check ───────────────────────────────────
if not hasattr(trainer, "fold_states"):
    raise RuntimeError("trainer.fold_states missing – ensure Trainer stores fold models")

pred_df = cross_val_predict(trainer, data)
assert {"Pred_Label", "Pred_Conf"}.issubset(pred_df.columns)

print(pred_df[["Pred_Label", "Pred_Conf"]].head())
print("✅ majority-vote smoke-test passed — rows:", len(pred_df))
