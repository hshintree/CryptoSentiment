#!/usr/bin/env python3
"""
Full 5-fold CV on **Ea** (year 2020, 36–60 k balanced tweets),
then out-of-sample evaluation on **Eb** (≈40 k event-sampled tweets
from 2015-2019 & 2021-2023, paper-exact configuration).

• strict temporal separation (Ea 2020 ⟂ Eb ≠2020)  
• per-fold preprocessing & EWMA-TBL labelling to avoid leakage  
• single "best-epoch" checkpoint per fold
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import os

from gpu_scripts.preprocessor import Preprocessor
from gpu_scripts.market_labeler_ewma import MarketLabelerTBL
from gpu_scripts.model import Model
from gpu_scripts.trainer import Trainer
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support)
from glob import glob

ap = argparse.ArgumentParser()
ap.add_argument("--no-save", action="store_true", help="Do not write CSV artifacts (signals_*.csv, *_daily_signals.csv)")
ap.add_argument("--timestamped", action="store_true", help="Append a timestamp suffix to CSV artifact filenames")
args = ap.parse_args()

# ── 1. Configuration ───────────────────────────────────────────
# ── paths ──────────────────────────────────────────────────────────
CFG        = "config.yaml"
EA_CSV     = Path("data/#1train.csv")          # Ea (training)
EB_CSV     = Path("data/#2val.csv")            # Eb (evaluation)

def strip_leaks(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leaky columns that would cause data leakage."""
    leaky = [
        "Upper Barrier", "Lower Barrier", "Vertical Barrier",
        "Volatility", "Previous Label", "Previous_Label"
    ]
    return df.drop(columns=[c for c in leaky if c in df.columns])

# ── 1.5. Load EA data (2020 training) ──────────────────────
print("="*70)
print("🚀 FULL CV RUN: EA TRAINING (2020) + EB EVALUATION (≠2020)")
print("="*70)

print("\n📁 Loading EA dataset  (#1train.csv  – 2020 only)…")
# parse_dates can silently fail → coerce to datetime explicitly
ea = pd.read_csv(EA_CSV)
ea = ea[pd.to_datetime(ea["date"], errors="coerce").notna()]  # drop NaT rows
ea = ea.assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
ea = strip_leaks(ea).rename(columns={"date": "Tweet Date"})
# guarantee the dtype
ea["Tweet Date"] = pd.to_datetime(ea["Tweet Date"], errors="coerce")
print(f"EA columns after strip: {ea.columns.tolist()}")
print(f"EA rows: {len(ea):,} tweets")

# ── 1.6  Load EB (event evaluation) ──────────────────────────
print("\n📁 Loading EB dataset  (#2val.csv – event evaluation)…")
eb = pd.read_csv(EB_CSV)
# date cleaning
eb = eb[pd.to_datetime(eb["date"], errors="coerce").notna()]
eb = eb.assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
eb = strip_leaks(eb).rename(columns={"date": "Tweet Date"})
eb["Tweet Date"] = pd.to_datetime(eb["Tweet Date"], errors="coerce")
print(f"EB  columns after strip: {eb.columns.tolist()}")
print(f"EB  rows: {len(eb):,} tweets")

# ── 1 bis.  EA-only indicator diagnostics (paper Table 3 style) ─────────
print("\n🔍 Paper-style indicator correlations **on EA (2020)**")

# ➊ technical indicators & buckets
pre_ea  = Preprocessor(CFG)
ea_proc = pre_ea.fit_transform(ea.copy())          # adds RSI_raw / ROC_raw

# ➋ daily aggregation (mean indicators, first label)
_code = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
daily_ea = (
    ea_proc
      .groupby(ea_proc["Tweet Date"].dt.normalize())
      .agg({"RSI_raw": "mean",
            "ROC_raw": "mean",
            "Label":   "first"})
      .replace(_code)
)

# ➌ Pearson ρ (numeric indicators ↔ label)
pear = daily_ea.corr().iloc[:-1, -1]     # each feature vs ground-truth label
for feat, rho in pear.items():
    print(f"   ρ({feat:14s} , Label) = {rho:+.3f}")

# ➍ Chi-square association (bucket ↔ label, 3 × 3 contingency)
from scipy.stats import chi2_contingency
import numpy as np

#   helper to bucketise into bearish/neutral/bullish (same as preprocessing)
def _tri_bucket(series, thresh_lo=-np.inf, thresh_hi=np.inf):
    """map numeric → {bearish,neutral,bullish}"""
    cats = np.where(series < 0, "bearish",
           np.where(series > 0, "bullish", "neutral"))
    return pd.Series(cats, index=series.index)

rsib = _tri_bucket(daily_ea["RSI_raw"] - 50)   # RSI centred at 50
rocb = _tri_bucket(daily_ea["ROC_raw"])
lbl  = daily_ea["Label"]

for name, buckets in [("RSI_bucket", rsib), ("ROC_bucket", rocb)]:
    tbl = pd.crosstab(buckets, lbl)
    chi2, p, _, _ = chi2_contingency(tbl)
    print(f"   χ²({name:11s}) = {chi2:6.1f}   p-value = {p:.4f}")

print("─────────────────────────────────────────────────────────────")

# Verify temporal separation
ea_years = sorted(ea['Tweet Date'].dt.year.unique())
eb_years  = sorted(eb['Tweet Date'].dt.year.unique())
print(f"\n🔍 Temporal verification:")
print(f"  EA years: {ea_years} (training)")
print(f"  EB  years: {eb_years} (evaluation)")

overlap = set(ea_years) & set(eb_years)
if overlap:
    print(f"❌ ERROR: Year overlap detected: {overlap}")
    exit(1)
else:
    print("✅ No temporal overlap - proper out-of-sample setup")

# ── 2. Check ground-truth labels are preserved ──────────────────
print(f"\n🔍 Ground-truth label verification:")
if 'Label' in ea.columns:
    ea_label_dist = ea['Label'].value_counts()
    print(f"  EA label distribution: {dict(ea_label_dist)}")
else:
    print("  ❌ EA missing 'Label' column!")

if 'Label' in eb.columns:
    eb_label_dist = eb['Label'].value_counts()
    print(f"  EB  label distribution:  {dict(eb_label_dist)}")
else:
    print("  ❌ EB missing 'Label' column!")

# ── 3. Instantiate model & trainer ─────────────────────────────
print(f"\n⚙️  Setting up model and trainer...")
with open(CFG) as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg["model"]

# Enable MPS fallback for better compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

model       = Model(model_cfg)
best_device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥️  Detected device: {best_device}")

# MPS-specific optimizations
if best_device == "mps":
    print("🚀 MPS detected - enabling optimizations:")
    print("   • Single-worker DataLoader to avoid pickle issues")
    print("   • File system sharing strategy")
    print("   • Forced float32 precision")
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_default_dtype(torch.float32)

trainer     = Trainer(model, ea, CFG, quiet=False)  # Trainer prints its own device

# quick knobs before trainer.train()
trainer.epochs        = 2          # ← from 2 → 3
trainer.learning_rate = 1e-5       
# If you switch to prompt-tuning / last-layer-only:
# trainer.learning_rate = 8e-5

print(f"\n💡 Training configuration:")
print(f"  Batch size: {trainer.batch_size}")
print(f"  Epochs: {trainer.epochs}")
print(f"  Learning rate: {trainer.learning_rate}")
print(f"  Prompt tuning: {model.use_prompt_tuning}")
print(f"\n📈 Expected progression:")
print(f"  Training loss: ~0.9 → 0.6 over 3 epochs")
print(f"  Validation accuracy: 45-55% (realistic for small Ea subset)")
print(f"  Each fold will have similar calendar length in train vs val")

# ── 4. Train on EA dataset ─────────────────────────────────────
print(f"\n" + "="*70)
print(f"🎯 STARTING 5-FOLD CV TRAINING ON EA (2020)")
print(f"="*70)
print(f"⚡ Training on {len(ea):,} EA tweets...")
print("💾 Memory Management: Models moved to CPU after each fold to free MPS memory")
print("🔒 Leak Prevention: Preprocessing and labeling done per-fold")
print("📊 Expected realistic metrics: 40-85% validation accuracy")
print("⚠️  If you see 95%+ accuracy, there's still label leakage!")

trainer.train()

# ── 5. Save per-fold checkpoints ───────────────────────────────
print(f"\n💾 Saving model checkpoints...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir  = Path("models") / f"fixed?_{timestamp}"
for fold_idx, fold_state in enumerate(trainer.fold_states, start=1):
    ckpt_dir = base_dir / f"fold{fold_idx}_epoch{trainer.epochs}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save the fine-tuned model & tokenizer for this fold (models are on CPU)
    fold_state["model"].bert_model.save_pretrained(ckpt_dir)
    fold_state["model"].tokenizer.save_pretrained(ckpt_dir)
    print(f"✓ Saved fold {fold_idx} checkpoint to {ckpt_dir}")

# ── 6. Predictions on the EB event-evaluation split ─────────────
print(f"\n" + "="*70)
print(f"🔮 OUT-OF-SAMPLE EVALUATION ON EB (events 2015-19 & 2021-23)")
print(f"="*70)

suffix = f"_{timestamp}" if args.timestamped else ""

if trainer.fold_states:
    print("🔮 Generating predictions on EB evaluation set…")
    
    print(f"  🎯 Running softmax-averaged ensemble on {len(eb):,} EB tweets…")
    sig_eb = trainer.ensemble_predict(eb, weighted=True)
    
    # Save per-tweet predictions
    if not args.no_save:
        sig_eb.to_csv(f"signals_eb{suffix}.csv", index=False)
        print(f"✓ Saved per-tweet predictions → signals_eb{suffix}.csv")
    
    # Generate daily signals via majority vote
    print("  📅 Aggregating daily signals...")
    daily_eb = (
        sig_eb.groupby("Tweet Date").Pred_Label
        .agg(lambda x: x.value_counts().idxmax())
        .sort_index()
        .rename("Daily_Signal")
    )
    if not args.no_save:
        daily_eb.to_csv(f"eb_daily_signals{suffix}.csv")
        print(f"✓ Saved per-day signals     → eb_daily_signals{suffix}.csv")
    
    # Summary statistics
    eb_pred_dist    = sig_eb['Pred_Label'].value_counts()
    daily_pred_dist = daily_eb.value_counts()
    print(f"\n📊 EB Prediction Summary:")
    print(f"  Per-tweet predictions: {dict(eb_pred_dist)}")
    print(f"  Daily signals:        {dict(daily_pred_dist)}")
    
    # Performance hint
    if 'Label' in sig_eb.columns:
        # --- core metrics -------------------------------------------------
        y_true = sig_eb['Label'].map({'Bearish':0,'Neutral':1,'Bullish':2}).values
        y_pred = sig_eb['Pred_Label'].map({'Bearish':0,'Neutral':1,'Bullish':2}).values

        acc  = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)

        print(f"  Accuracy : {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall   : {rec:.3f}")
        print(f"  F1-score : {f1:.3f}")
        
        if acc > 0.90:
            print("  ⚠️  Very high accuracy - check for data leakage!")
        elif acc > 0.60:
            print("  ✅ Good performance on out-of-sample data")
        else:
            print("  📈 Moderate performance - may need model improvements")
    
    # ── 6.5. ALSO test on EA (in-sample) ─────────────────────────────
    print(f"\n" + "="*70)
    print(f"🔮 IN-SAMPLE EVALUATION ON EA (2020 training data)")
    print(f"="*70)
    
    print(f"  🎯 Running softmax-averaged ensemble on {len(ea):,} EA tweets…")
    sig_ea = trainer.ensemble_predict(ea, weighted=True)
    
    # Save per-tweet predictions for EA
    if not args.no_save:
        sig_ea.to_csv(f"signals_ea{suffix}.csv", index=False)
        print(f"✓ Saved per-tweet predictions → signals_ea{suffix}.csv")
    
    # Generate daily signals via majority vote for EA
    print("  📅 Aggregating daily signals...")
    daily_ea = (
        sig_ea.groupby("Tweet Date").Pred_Label
        .agg(lambda x: x.value_counts().idxmax())
        .sort_index()
        .rename("Daily_Signal")
    )
    if not args.no_save:
        daily_ea.to_csv(f"ea_daily_signals{suffix}.csv")
        print(f"✓ Saved per-day signals     → ea_daily_signals{suffix}.csv")
    
    # Summary statistics for EA
    ea_pred_dist    = sig_ea['Pred_Label'].value_counts()
    daily_ea_pred_dist = daily_ea.value_counts()
    print(f"\n📊 EA Prediction Summary:")
    print(f"  Per-tweet predictions: {dict(ea_pred_dist)}")
    print(f"  Daily signals:        {dict(daily_ea_pred_dist)}")
    
    # Performance hint for EA
    if 'Label' in sig_ea.columns:
        # --- core metrics -------------------------------------------------
        y_true = sig_ea['Label'].map({'Bearish':0,'Neutral':1,'Bullish':2}).values
        y_pred = sig_ea['Pred_Label'].map({'Bearish':0,'Neutral':1,'Bullish':2}).values

        acc  = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)

        print(f"  Accuracy : {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall   : {rec:.3f}")
        print(f"  F1-score : {f1:.3f}")
        
        if acc > 0.90:
            print("  ⚠️  Very high accuracy - check for data leakage!")
        elif acc > 0.60:
            print("  ✅ Good performance on in-sample data")
        else:
            print("  📈 Moderate performance - may need model improvements")
else:
    print("⚠️  No trained models – skipping EB inference")

print(f"\n" + "="*70)
print(f"🎉 FULL CV RUN COMPLETED!")
print(f"="*70)
print(f"📁 Files generated:")
print(f"  • Model checkpoints: models/ea_2019-2020_{timestamp}/")
if not args.no_save:
    print(f"  • EA  predictions: signals_ea{suffix}.csv")
    print(f"  • EA  daily signals: ea_daily_signals{suffix}.csv")
    print(f"  • EB  predictions: signals_eb{suffix}.csv")
    print(f"  • EB  daily signals: eb_daily_signals{suffix}.csv")
print(f"  • Training metrics: training_metrics.json")
print(f"\n✅ Ready for paper results analysis!") 