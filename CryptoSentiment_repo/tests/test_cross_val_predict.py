#!/usr/bin/env python3
"""
Tiny smoketest for cross_val_predict function.
Just verifies the function runs without crashing.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from gpu_scripts.model import Model
from gpu_scripts.trainer import Trainer, cross_val_predict

def test_cross_val_predict():
    print("🧪 CROSS_VAL_PREDICT SMOKETEST")
    print("=" * 40)
    
    # ── 1. Create minimal synthetic data ──────────────────────
    print("📝 Creating minimal synthetic data...")
    
    # Create 150 tweets over 30 days (enough for CV with 13-day gap)
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    data = []
    
    for date in dates:
        for i in range(5):  # 5 tweets per day
            data.append({
                "Tweet Date": date,
                "Tweet Content": f"bitcoin sample tweet {i} on {date.date()}",
                "Close": 10000 + np.random.normal(0, 100),  # price around $10k
                "Volume": 1000000 + np.random.normal(0, 100000),
                "Label": np.random.choice(["Bearish", "Neutral", "Bullish"]),
                "RSI": 50 + np.random.normal(0, 10),
                "ROC": np.random.normal(0, 2),
                "Previous Label": "Neutral"  # simple default
            })
    
    df = pd.DataFrame(data)
    print(f"✓ Created {len(df)} synthetic tweets over {len(dates)} days")
    print(f"  Date range: {df['Tweet Date'].min().date()} to {df['Tweet Date'].max().date()}")
    
    # ── 2. Setup minimal trainer ──────────────────────────────
    print("⚙️  Setting up minimal trainer...")
    
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    model = Model(cfg["model"])
    trainer = Trainer(model, df, "config.yaml")
    
    # Ultra-minimal settings
    trainer.epochs = 1
    trainer.batch_size = 4
    
    print(f"✓ Trainer setup complete")
    
    # ── 3. Do minimal training to create fold_states ─────────
    print("🚀 Running minimal training (1 epoch)...")
    
    trainer.train()
    
    print(f"✓ Training complete, {len(trainer.fold_states)} fold states created")
    
    # Check if any folds were actually created
    if len(trainer.fold_states) == 0:
        print("❌ No fold states created - data too small for 13-day gap CV")
        print("💡 Need more days of data for blocked purged CV to work")
        return
    
    # ── 4. Test cross_val_predict ─────────────────────────────
    print("🔮 Testing cross_val_predict...")
    
    # Test on same data (just to verify function works)
    result = cross_val_predict(trainer, df)
    
    print(f"✓ cross_val_predict completed!")
    print(f"  Result shape: {result.shape}")
    print(f"  Result columns: {list(result.columns)}")
    
    # Check if we got predictions
    pred_cols = [col for col in result.columns if 'Pred' in col]
    print(f"  Prediction columns: {pred_cols}")
    
    if 'Pred_Label' in result.columns:
        pred_dist = result['Pred_Label'].value_counts()
        print(f"  Predictions: {dict(pred_dist)}")
    
    # Quick sample
    if len(result) > 0:
        sample_cols = ['Tweet Date', 'Label', 'Pred_Label', 'Pred_Conf']
        available_cols = [c for c in sample_cols if c in result.columns]
        sample = result[available_cols].head(3)
        print(f"\n📋 Sample results:")
        print(sample.to_string(index=False))
    
    print(f"\n✅ CROSS_VAL_PREDICT SMOKETEST PASSED!")
    print("🚀 Function runs without errors")

if __name__ == "__main__":
    test_cross_val_predict() 