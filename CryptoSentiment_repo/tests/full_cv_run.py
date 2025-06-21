#!/usr/bin/env python3
"""
Full 5-fold CV run on EA dataset (2019-2020), then stress-test on VAL dataset (2021).
Uses proper temporal separation and prevents data leakage.
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

from preprocessor import Preprocessor
from market_labeler_ewma import MarketLabelerTBL
from model import Model
from trainer import Trainer, cross_val_predict
from glob import glob

# ── 1. Configuration ───────────────────────────────────────────
CFG     = "config.yaml"
EA_CSV  = max(glob("data/ea_dataset_2019-2020_train_*.csv"))
VAL_CSV = max(glob("data/val_dataset_2021_stress_*.csv"))

def strip_leaks(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leaky columns that would cause data leakage."""
    leaky = [
        "Upper Barrier", "Lower Barrier", "Vertical Barrier",
        "Volatility", "Previous Label", "Previous_Label"
    ]
    return df.drop(columns=[c for c in leaky if c in df.columns])

# ── 1.5. Load EA data (2019-2020 training) ──────────────────────
print("="*70)
print("🚀 FULL CV RUN: EA TRAINING (2019-2020) + VAL STRESS-TEST (2021)")
print("="*70)

print("\n📁 Loading EA dataset (2019-2020 training data)...")
# parse_dates can silently fail → coerce to datetime explicitly
ea = pd.read_csv(EA_CSV)
ea = ea[pd.to_datetime(ea["date"], errors="coerce").notna()]  # drop NaT rows
ea = ea.assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
ea = strip_leaks(ea).rename(columns={"date": "Tweet Date"})
# guarantee the dtype
ea["Tweet Date"] = pd.to_datetime(ea["Tweet Date"], errors="coerce")
print(f"EA columns after strip: {ea.columns.tolist()}")
print(f"EA rows: {len(ea):,} tweets")

# ── 1.6  Load VAL (2021) ────────────────────────────────────
print("\n📁 Loading VAL dataset (2021 stress-test data)...")
val = pd.read_csv(VAL_CSV)
val = val[pd.to_datetime(val["date"], errors="coerce").notna()]  # drop NaT rows
val = val.assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
val = strip_leaks(val).rename(columns={"date": "Tweet Date"})
val["Tweet Date"] = pd.to_datetime(val["Tweet Date"], errors="coerce")
print(f"VAL columns after strip: {val.columns.tolist()}")
print(f"VAL rows: {len(val):,} tweets")

# Verify temporal separation
ea_years = sorted(ea['Tweet Date'].dt.year.unique())
val_years = sorted(val['Tweet Date'].dt.year.unique())
print(f"\n🔍 Temporal verification:")
print(f"  EA years: {ea_years} (training)")
print(f"  VAL years: {val_years} (stress-test)")

overlap = set(ea_years) & set(val_years)
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

if 'Label' in val.columns:
    val_label_dist = val['Label'].value_counts()
    print(f"  VAL label distribution: {dict(val_label_dist)}")
else:
    print("  ❌ VAL missing 'Label' column!")

# ── 3. Instantiate model & trainer ─────────────────────────────
print(f"\n⚙️  Setting up model and trainer...")
with open(CFG) as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg["model"]
model     = Model(model_cfg)
trainer   = Trainer(model, ea, CFG)  # Pass EA data, preprocessing happens per-fold
print(f"Using device: {trainer.device}")

# Override hyperparams
trainer.batch_size    = 12
trainer.epochs        = 2
trainer.learning_rate = 1e-5

print(f"\n💡 Training configuration:")
print(f"  Batch size: {trainer.batch_size}")
print(f"  Epochs: {trainer.epochs}")
print(f"  Learning rate: {trainer.learning_rate}")
print(f"  Prompt tuning: {model.use_prompt_tuning}")

# ── 4. Train on EA dataset ─────────────────────────────────────
print(f"\n" + "="*70)
print(f"🎯 STARTING 5-FOLD CV TRAINING ON EA (2019-2020)")
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
base_dir  = Path("models") / f"ea_2019-2020_{timestamp}"
for fold_idx, fold_state in enumerate(trainer.fold_states, start=1):
    ckpt_dir = base_dir / f"fold{fold_idx}_epoch{trainer.epochs}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save the fine-tuned model & tokenizer for this fold (models are on CPU)
    fold_state["model"].bert_model.save_pretrained(ckpt_dir)
    fold_state["model"].tokenizer.save_pretrained(ckpt_dir)
    print(f"✓ Saved fold {fold_idx} checkpoint to {ckpt_dir}")

# ── 6. Predictions on the 2021 stress-test (VAL) split ──────────
print(f"\n" + "="*70)
print(f"🔮 STRESS-TEST EVALUATION ON VAL (2021)")
print(f"="*70)

if trainer.fold_states:
    print("🔮 Generating predictions on 2021 stress-test set…")
    
    # Preprocess and label VAL data for prediction
    print("  📊 Preprocessing VAL data...")
    from preprocessor import Preprocessor
    pred_preprocessor = Preprocessor(CFG)
    pred_preprocessor.fit(ea)           # fit on training only
    val_prep = pred_preprocessor.transform(val.copy())
    
    print("  🏷️  Labeling VAL data...")
    pred_labeler = MarketLabelerTBL(CFG)
    val_prep = pred_labeler.label_data(val_prep)
    val_prep = val_prep.rename(columns={"date": "Tweet Date"})
    
    print(f"  🎯 Running cross-validation prediction on {len(val_prep):,} VAL tweets...")
    sig_val = cross_val_predict(trainer, val_prep)
    
    # Save per-tweet predictions
    sig_val.to_csv("signals_val_2021.csv", index=False)
    print("✓ Saved per-tweet predictions → signals_val_2021.csv")
    
    # Generate daily signals via majority vote
    print("  📅 Aggregating daily signals...")
    daily = (
        sig_val.groupby("Tweet Date").Pred_Label
        .agg(lambda x: x.value_counts().idxmax())
        .sort_index()
        .rename("Daily_Signal")
    )
    daily.to_csv("val_daily_signals.csv")
    print("✓ Saved per-day signals     → val_daily_signals.csv")
    
    # Summary statistics
    val_pred_dist = sig_val['Pred_Label'].value_counts()
    daily_pred_dist = daily.value_counts()
    print(f"\n📊 VAL Prediction Summary:")
    print(f"  Per-tweet predictions: {dict(val_pred_dist)}")
    print(f"  Daily signals: {dict(daily_pred_dist)}")
    
    # Performance hint
    if 'Label' in sig_val.columns:
        accuracy = (sig_val['Pred_Label'] == sig_val['Label']).mean()
        print(f"  Raw accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        if accuracy > 0.90:
            print("  ⚠️  Very high accuracy - check for data leakage!")
        elif accuracy > 0.60:
            print("  ✅ Good performance on stress-test data")
        else:
            print("  📈 Moderate performance - may need model improvements")
else:
    print("⚠️  No trained models – skipping VAL inference")

print(f"\n" + "="*70)
print(f"🎉 FULL CV RUN COMPLETED!")
print(f"="*70)
print(f"📁 Files generated:")
print(f"  • Model checkpoints: models/ea_2019-2020_{timestamp}/")
print(f"  • VAL predictions: signals_val_2021.csv")
print(f"  • Daily signals: val_daily_signals.csv")
print(f"  • Training metrics: training_metrics.json")
print(f"\n✅ Ready for paper results analysis!") 