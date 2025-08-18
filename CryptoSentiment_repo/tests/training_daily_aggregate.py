#!/usr/bin/env python3
"""
Smoke-test daily aggregation.

• uses the real config.yaml — nothing is written to disk
• trains on 300 tweets, 1 epoch, batch=4
• verifies Pred_Label & Pred_Conf appear after cross-fold voting
"""

import yaml, torch
import pandas as pd
from pathlib import Path
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
preds = cross_val_predict(trainer, data)

    # 3) daily aggregations
daily_label = preds.groupby("Tweet Date")["Pred_Label"] \
                       .agg(lambda s: s.value_counts().idxmax())
daily_conf  = preds.groupby("Tweet Date")["Pred_Conf"].mean()
daily = pd.concat([daily_label, daily_conf], axis=1).reset_index()

    # 4) sanity checks
unique_days = data["Tweet Date"].nunique()
assert daily.shape[0] == unique_days
assert set(daily.columns) == {"Tweet Date", "Pred_Label", "Pred_Conf"}

# optional: save and inspect
tmp_path = Path("tmp")
tmp_path.mkdir(parents=True, exist_ok=True)
daily.to_csv(tmp_path / "daily.csv", index=False)
print(f"✅ wrote {len(daily):,} rows → {tmp_path / 'daily.csv'}")