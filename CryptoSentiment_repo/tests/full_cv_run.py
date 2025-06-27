#!/usr/bin/env python3
"""
Full 5-fold CV on **Ea** (year 2020, 36–60 k balanced tweets),
then out-of-sample evaluation on **Eb** (≈40 k event-sampled tweets
from 2015-2019 & 2021-2023, paper-exact configuration).

• strict temporal separation (Ea 2020 ⟂ Eb ≠2020)  
• per-fold preprocessing & EWMA-TBL labelling to avoid leakage  
• single "best-epoch" checkpoint per fold
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

from preprocessor import Preprocessor
from market_labeler_ewma import MarketLabelerTBL
from model import Model
from trainer import Trainer, cross_val_predict
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support)
from glob import glob

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
model     = Model(model_cfg)
trainer   = Trainer(model, ea, CFG)  # Pass EA data, preprocessing happens per-fold
print(f"Using device: {trainer.device}")

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

if trainer.fold_states:
    print("🔮 Generating predictions on EB evaluation set…")
    
    # Preprocess and label EB data for prediction
    print("  📊 Preprocessing EB data...")
    from preprocessor import Preprocessor
    pred_preprocessor = Preprocessor(CFG)
    pred_preprocessor.fit(ea)           # fit on training only
    eb_prep  = pred_preprocessor.transform(eb.copy())
    
    # ⚠️  Do NOT relabel Eb – keep its ground-truth labels.
    eb_prep  = eb_prep.rename(columns={"date": "Tweet Date"})
    # add causal Previous-Label feature once (uses past values only)
    from market_labeler_ewma import MarketFeatureGenerator
    feat_gen = MarketFeatureGenerator(CFG)
    feat_gen.fit(eb_prep)
    eb_prep["Previous Label"] = feat_gen.transform(eb_prep)
    
    print(f"  🎯 Running cross-validation prediction on {len(eb_prep):,} EB tweets…")
    sig_eb = cross_val_predict(trainer, eb_prep)
    
    # Save per-tweet predictions
    sig_eb.to_csv("signals_eb.csv", index=False)
    print("✓ Saved per-tweet predictions → signals_eb.csv")
    
    # Generate daily signals via majority vote
    print("  📅 Aggregating daily signals...")
    daily_eb = (
        sig_eb.groupby("Tweet Date").Pred_Label
        .agg(lambda x: x.value_counts().idxmax())
        .sort_index()
        .rename("Daily_Signal")
    )
    daily_eb.to_csv("eb_daily_signals.csv")
    print("✓ Saved per-day signals     → eb_daily_signals.csv")
    
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

        # --------- quick leakage sanity check ----------------------------
        if acc > 0.90:
            print("  ⚠️  Very high accuracy - check for data leakage!")
        elif acc > 0.60:
            print("  ✅ Good performance on out-of-sample data")
        else:
            print("  📈 Moderate performance - may need model improvements")
else:
    print("⚠️  No trained models – skipping EB inference")

print(f"\n" + "="*70)
print(f"🎉 FULL CV RUN COMPLETED!")
print(f"="*70)
print(f"📁 Files generated:")
print(f"  • Model checkpoints: models/ea_2019-2020_{timestamp}/")
print(f"  • EB  predictions: signals_eb.csv")
print(f"  • Daily signals:   eb_daily_signals.csv")
print(f"  • Training metrics: training_metrics.json")
print(f"\n✅ Ready for paper results analysis!") 