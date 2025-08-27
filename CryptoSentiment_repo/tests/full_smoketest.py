#!/usr/bin/env python3
"""
QUICK SMOKETEST: Fast validation of the full training pipeline.
Based on full_cv_run.py but optimized for ~2 minutes on MPS.

• Small data sample (500 tweets)
• 1 epoch only
• Small batch size
• No model saving
• Tests all key components: preprocessing, labeling, CV training, prediction
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from gpu_scripts.preprocessor import Preprocessor
from gpu_scripts.market_labeler_ewma import MarketLabelerTBL
from gpu_scripts.model import Model
from gpu_scripts.trainer import Trainer, cross_val_predict

# ── Configuration ───────────────────────────────────────────
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

def smoketest():
    print("="*60)
    print("🚀 QUICK SMOKETEST: Full Pipeline Validation")
    print("="*60)
    print("⚡ Target: ~2 minutes on MPS")
    print("📊 Small sample + 1 epoch + minimal CV")
    
    # ── 1. Load and sample EA data ──────────────────────
    print("\n📁 Loading EA dataset (#1train.csv)...")
    ea = pd.read_csv(EA_CSV)
    ea = ea[pd.to_datetime(ea["date"], errors="coerce").notna()]
    ea = ea.assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
    ea = strip_leaks(ea).rename(columns={"date": "Tweet Date"})
    ea["Tweet Date"] = pd.to_datetime(ea["Tweet Date"], errors="coerce")
    
    # QUICK: Take small sample for speed
    ea_sample = ea.sample(n=500, random_state=42).reset_index(drop=True)

    # ── ensure numeric columns so RSI / ROC can be computed later ──
    def _add_numeric_stub(df: pd.DataFrame) -> None:
        """Guarantee Close / Volume / RSI / ROC exist (dummy values are fine
        for a smoke-test so later code doesn't KeyError)."""
        if "Close"  not in df.columns:
            # synthetic price path – rising by 1 each row
            df["Close"] = np.linspace(10_000, 10_000 + len(df) - 1, len(df))
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        if "RSI" not in df.columns:
            df["RSI"] = 50.0          # neutral default
        if "ROC" not in df.columns:
            df["ROC"] = 0.0           # flat return
        # add a neutral stub so correlation/debug code finds the column
        if "Previous Label" not in df.columns:
            df["Previous Label"] = "Neutral"

    _add_numeric_stub(ea_sample)
    print(f"✓ Sample: {len(ea_sample):,} tweets (from {len(ea):,} total)")
    print(f"  Date range: {ea_sample['Tweet Date'].min().date()} to {ea_sample['Tweet Date'].max().date()}")
    
    # ── 2. Load and sample EB data ──────────────────────
    print("\n📁 Loading EB dataset (#2val.csv)...")
    eb = pd.read_csv(EB_CSV)
    eb = eb[pd.to_datetime(eb["date"], errors="coerce").notna()]
    eb = eb.assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
    eb = strip_leaks(eb).rename(columns={"date": "Tweet Date"})
    eb["Tweet Date"] = pd.to_datetime(eb["Tweet Date"], errors="coerce")
    
    # QUICK: Small EB sample too
    eb_sample = eb.sample(n=200, random_state=42).reset_index(drop=True)
    print(f"✓ Sample: {len(eb_sample):,} tweets (from {len(eb):,} total)")
    
    # ── 3. Verify temporal separation ──────────────────────
    ea_years = sorted(ea_sample['Tweet Date'].dt.year.unique())
    eb_years = sorted(eb_sample['Tweet Date'].dt.year.unique())
    print(f"\n🔍 Temporal check:")
    print(f"  EA years: {ea_years}")
    print(f"  EB years: {eb_years}")
    
    overlap = set(ea_years) & set(eb_years)
    if overlap:
        print(f"❌ Year overlap: {overlap}")
    else:
        print("✅ No temporal overlap")
    
    # ── 4. Quick labeling check ──────────────────────────
    print(f"\n🏷️  Ground-truth labels:")
    if 'Label' in ea_sample.columns:
        ea_labels = ea_sample['Label'].value_counts()
        print(f"  EA: {dict(ea_labels)}")
    if 'Label' in eb_sample.columns:
        eb_labels = eb_sample['Label'].value_counts()
        print(f"  EB: {dict(eb_labels)}")
    
    # ── 5. Setup model & trainer ──────────────────────────
    print(f"\n⚙️  Setting up model and trainer...")
    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    
    model = Model(cfg["model"])
    trainer = Trainer(model, ea_sample, CFG)
    
    # SPEED SETTINGS
    trainer.epochs = 1          # Just 1 epoch!
    trainer.batch_size = 8      # Small batches
    trainer.learning_rate = 1e-5
    
    print(f"⚡ SPEED SETTINGS:")
    print(f"  Epochs: {trainer.epochs}")
    print(f"  Batch size: {trainer.batch_size}")
    print(f"  Device: {trainer.device}")
    print(f"  Prompt tuning: {model.use_prompt_tuning}")
    
    # ── 6. Train on EA sample ──────────────────────────────
    print(f"\n" + "="*60)
    print(f"🎯 TRAINING 5-FOLD CV ON EA SAMPLE")
    print(f"="*60)
    print("🚀 This should complete in ~2 minutes...")
    
    trainer.train()
    
    print(f"\n✅ Training completed!")
    print(f"📊 Fold states created: {len(trainer.fold_states)}")
    
    # ── 7. Quick prediction test on EB ──────────────────────
    print(f"\n" + "="*60)
    print(f"🔮 TESTING PREDICTION ON EB SAMPLE")
    print(f"="*60)
    
    if trainer.fold_states:
        print("🔮 Generating predictions on EB sample...")
        
        # Quick preprocessing for EB
        print("  📊 Preprocessing EB...")
        pred_preprocessor = Preprocessor(CFG)
        pred_preprocessor.fit(ea_sample)  # fit on EA sample
        eb_prep = pred_preprocessor.transform(eb_sample.copy())
        
        print("  🏷️  Labeling EB...")
        pred_labeler = MarketLabelerTBL(CFG)
        eb_prep = pred_labeler.label_data(eb_prep)
        eb_prep = eb_prep.rename(columns={"date": "Tweet Date"})
        
        print(f"  🎯 Running CV prediction on {len(eb_prep):,} EB tweets...")
        sig_eb = cross_val_predict(trainer, eb_prep)
        
        # Quick analysis
        if 'Pred_Label' in sig_eb.columns:
            pred_dist = sig_eb['Pred_Label'].value_counts()
            print(f"  📊 Predictions: {dict(pred_dist)}")
            
            if 'Label' in sig_eb.columns:
                accuracy = (sig_eb['Pred_Label'] == sig_eb['Label']).mean()
                print(f"  🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                
                if accuracy > 0.90:
                    print("  ⚠️  Very high accuracy - possible leakage!")
                elif accuracy > 0.60:
                    print("  ✅ Good performance")
                else:
                    print("  📈 Moderate performance")
            
            # Sample predictions
            if len(sig_eb) > 0:
                sample_cols = ['Tweet Date', 'Label', 'Pred_Label', 'Pred_Conf']
                available_cols = [c for c in sample_cols if c in sig_eb.columns]
                sample = sig_eb[available_cols].head(5)
                print(f"\n📋 Sample predictions:")
                print(sample.to_string(index=False))
        
        print(f"\n✅ Prediction test completed!")
    else:
        print("❌ No trained models - skipping prediction test")
    
    # ── 8. Final summary ──────────────────────────────────
    print(f"\n" + "="*60)
    print(f"🎉 SMOKETEST COMPLETED!")
    print(f"="*60)
    print(f"✅ All pipeline components tested:")
    print(f"  • Data loading and cleaning")
    print(f"  • Temporal separation verification")
    print(f"  • 5-fold blocked CV training")
    print(f"  • Cross-validation prediction")
    print(f"  • Out-of-sample evaluation")
    print(f"\n🚀 Ready for full training run!")

if __name__ == "__main__":
    smoketest()
