#!/usr/bin/env python3
"""
Full 5-fold CV run on EA dataset, using MPS device, temporal grouping fix and checkpoint saving.
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

from preprocessor import Preprocessor
from market_labeler_ewma import MarketLabelerEWMA
from model import Model
from trainer import Trainer, cross_val_predict

# ── 1. Configuration ───────────────────────────────────────────
CFG    = "config.yaml"
EA_CSV = "data/ea_dataset_2020_training_20250619_023215.csv"

# ── 1.5. Load data (NO preprocessing to prevent scaling leakage) ──
ea = pd.read_csv(EA_CSV, parse_dates=["date"])

# ── 2. Check what columns we actually have ───────────────────────────
print("Original columns:", ea.columns.tolist())

# ── 3. Define exactly the leaky columns to remove ────────────────────
leaky = [
    "Upper Barrier",
    "Lower Barrier", 
    "Vertical Barrier",
    "Volatility",
    "Label",
    "Previous Label",    # with space 
    "Previous_Label",    # with underscore (from original CSV)
]
to_drop = [c for c in leaky if c in ea.columns]
print("Dropping leaky columns:", to_drop)

# ── 4. Drop them ─────────────────────────────────────────────────────
ea = ea.drop(columns=to_drop)

# ── 5. Rename `date` → `Tweet Date` for Trainer compatibility ─────────
ea = ea.rename(columns={"date": "Tweet Date"})
print("Columns just before Trainer:", ea.columns.tolist())
# ✅ NO GLOBAL PREPROCESSING OR LABELING - this would cause data leakage!
# Both preprocessing and labeling will be done per-fold in the trainer
print("Skipping global preprocessing and labeling to prevent data leakage...")
print("  - Preprocessing: MinMaxScaler will be fitted per-fold")
print("  - Labeling: EWMA thresholds will be fitted per-fold")

# ── 3. Instantiate model & trainer ─────────────────────────────
with open(CFG) as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg["model"]
model     = Model(model_cfg)
trainer   = Trainer(model, ea, CFG)  # Pass raw data, preprocessing happens per-fold
print(f"Using device: {trainer.device}")

# Override hyperparams
trainer.batch_size    = 12
trainer.epochs        = 2
trainer.learning_rate = 1e-5

# ── 4. Train ───────────────────────────────────────────────────
print(f"⚡ Starting 5-fold CV training on EA ({len(ea)} tweets)…")
print("💾 Memory Management: Models moved to CPU after each fold to free MPS memory")
print("Expected realistic metrics: 40-85% validation accuracy")
print("If you see 95%+ accuracy, there's still label leakage!")
trainer.train()

# ── 5. Save per-fold checkpoints (after epoch 2) ───────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir  = Path("models") / timestamp
for fold_idx, fold_state in enumerate(trainer.fold_states, start=1):
    ckpt_dir = base_dir / f"fold{fold_idx}_epoch2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save the fine-tuned model & tokenizer for this fold (models are on CPU)
    fold_state["model"].bert_model.save_pretrained(ckpt_dir)
    fold_state["model"].tokenizer.save_pretrained(ckpt_dir)
    print(f"✓ Saved fold {fold_idx} checkpoint to {ckpt_dir}")

# ── 6. Final cross-validation predictions ──────────────────────
if hasattr(trainer, "fold_states") and len(trainer.fold_states) > 0:
    try:
        # For prediction, we need preprocessed and labeled data 
        # This is just for prediction consistency, not for training
        print("Preprocessing and labeling full dataset for prediction...")
        from preprocessor import Preprocessor
        pred_preprocessor = Preprocessor(CFG)
        ea_for_prediction = pred_preprocessor.preprocess(ea)  # Global preprocessing for prediction only
        
        pred_labeler = MarketLabelerEWMA("config_ewma.yaml")
        ea_for_prediction = pred_labeler.label_data(ea_for_prediction)
        ea_for_prediction = ea_for_prediction.rename(columns={"date": "Tweet Date"})
        
        sig_df = cross_val_predict(trainer, ea_for_prediction)
        out_file = "signals_per_tweet.csv"
        sig_df.to_csv(out_file, index=False)
        print(f"✓ Saved signals per tweet to {out_file}")
        
        # ── 7. (Optional) Daily aggregation ────────────────────────────
        daily = (
            sig_df
            .groupby("Tweet Date")
            .Pred_Label
            .agg(lambda x: x.value_counts().idxmax())
            .rename("Daily_Signal")
        )
        daily_file = "ea_daily_signals.csv"
        daily.to_csv(daily_file, index=True)
        print(f"✓ Saved daily signals to {daily_file}")
        
    except Exception as e:
        print(f"⚠️  Cross-validation prediction failed: {e}")
else:
    print("⚠️  No trained models available for prediction")

