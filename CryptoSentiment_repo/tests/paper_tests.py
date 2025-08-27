#!/usr/bin/env python3
"""
Script to fine-tune the Context-Aware (CA) CryptoBERT model
on your EA dataset with 5-fold grouped CV, exactly as in the paper.
"""

import argparse
import os
import yaml
import pandas as pd
from gpu_scripts.model       import Model
from gpu_scripts.trainer     import Trainer, cross_val_predict          # :contentReference[oaicite:0]{index=0}
from gpu_scripts.preprocessor import Preprocessor
from gpu_scripts.market_labeler_ewma import MarketLabelerTBL         # your enhanced labeler
from pathlib import Path
from datetime import datetime
import time, datetime

ap = argparse.ArgumentParser()
ap.add_argument("--no-save", action="store_true", help="Do not write CSV artifacts (signals_per_tweet.csv, ea_daily_signals.csv)")
ap.add_argument("--timestamped", action="store_true", help="Append timestamp to output filenames")
args = ap.parse_args()

# ── 1. Configuration ───────────────────────────────────────────
CFG = "config.yaml"
EA_CSV = "data/ea_dataset_2020_training_20250619_023215.csv"

# ── 2. Load & preprocess ──────────────────────────────────────
# Load raw EA CSV (tweets + dates + Close, etc.)
ea = pd.read_csv(EA_CSV, parse_dates=["date"])
ea = ea.rename(columns={"date": "Tweet Date"})
# Keep original column names for now - MarketLabelerTBL expects 'date' column

# Text cleaning, token normalization, feature extraction (ROC, RSI, lemmatization, etc.)
pp = Preprocessor(CFG)
ea_pp = pp.preprocess(ea)   # adds Tweet Content normalization, ROC, RSI, etc.

# Market-derived labels + EWMA volatility features (needs 'date' column name)
ml = MarketLabelerTBL("config.yaml")
ea_lbl = ml.label_data(ea_pp)  # adds 'Label' and 'Volatility' columns

# NOW rename columns for compatibility with trainer
ea_lbl = ea_lbl.rename(columns={
    "date": "Tweet Date",
    "Tweet Content": "Tweet Content"   # Keep if already correct, or rename from 'text' if needed
})

# The trainer will handle adding "Previous Label" feature automatically

# ── 3. Build & train model ────────────────────────────────────
# Instantiate the BERT model wrapper
with open(CFG) as f:
    model_cfg = yaml.safe_load(f)["model"]
model = Model(model_cfg)      # :contentReference[oaicite:1]{index=1}

# Create the CV trainer
trainer = Trainer(model, ea_lbl, CFG)

# (Optional) override defaults to match paper hyper-params
trainer.batch_size    = 12
trainer.epochs        = 2
trainer.learning_rate = 1e-5

print(f"⚡ Starting 5-fold CV training on EA ({len(ea_lbl)} tweets)…")
trainer.train()

# after trainer.train()
output_dir = Path("models") / datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print(f"{datetime.datetime.now().isoformat()} → TRAIN DONE, entering save+eval")

# 1) model save
t1 = time.time()
model.bert_model.save_pretrained(output_dir)
print("model.save_pretrained:", time.time() - t1, "s")

t2 = time.time()
model.tokenizer.save_pretrained(output_dir)
print("tokenizer.save_pretrained:", time.time() - t2, "s")

# 2) cross-val predict
t3 = time.time()
sig_df = cross_val_predict(trainer, ea_lbl, CFG)
print("cross_val_predict:", time.time() - t3, "s")

# 3) CSV writes
ts_suffix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamped else ""
if not args.no_save:
    sig_df.to_csv(f"signals_per_tweet{ts_suffix}.csv", index=False)
    print("to_csv:", time.time() - t4, "s")

# ── 5. (Optional) Daily aggregation ───────────────────────────
# Majority-vote per date to get one signal/day
daily = (
    sig_df
    .groupby("Tweet Date")
    .Pred_Label
    .agg(lambda x: x.value_counts().idxmax())
    .rename("Daily_Signal")
)
if not args.no_save:
    daily.to_csv(f"ea_daily_signals{ts_suffix}.csv")
    print("✓ Saved ea_daily_signals.csv")
