#!/usr/bin/env python3
"""
Updated cross-fold validation test with temporal grouping fix.

• Uses direct CSV loading like quick_train
• Tests the fixed Trainer with proper temporal CV splits
• Should show realistic performance metrics (not 99%+ accuracy)
"""

import yaml
import torch
import pandas as pd
from preprocessor import Preprocessor
from market_labeler_ewma import MarketLabelerTBL
from model import Model
from trainer import Trainer, cross_val_predict

def test_temporal_grouping():
    print("🧪 TEMPORAL GROUPING TEST")
    print("=" * 40)
    
    # Load data directly from CSV (like quick_train)
    print("📝 Loading and sampling data...")
    ea = pd.read_csv("data/ea_dataset_2020_training_20250619_023215.csv",
                     parse_dates=["date"])
    ea = ea.sample(n=5000, random_state=42).reset_index(drop=True)  # Reasonable sample
    ea = ea.rename(columns={"date": "Tweet Date"})
    
    print(f"Sample size: {len(ea)} tweets")
    print(f"Date range: {ea['Tweet Date'].min()} to {ea['Tweet Date'].max()}")
    
    # Preprocess the sample data
    print("🔧 Preprocessing data...")
    pp = Preprocessor("config.yaml")
    ea_processed = pp.preprocess(ea)
    
    # Apply labeling
    print("🏷️  Applying EWMA labeling...")
    ml = MarketLabelerTBL("config.yaml")
    ea_labeled = ml.label_data(ea_processed)
    
    # Rename for trainer compatibility
    ea_labeled = ea_labeled.rename(columns={"date": "Tweet Date"})
    
    # Note: Previous Label will be computed per-fold to prevent leakage
    
    print(f"Final data: {len(ea_labeled)} tweets")
    print(f"Label distribution:\n{ea_labeled['Label'].value_counts()}")
    
    # Test the new temporal grouping
    print("\n🔄 Testing temporal grouping...")
    # Create a temporary model just for testing grouping
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    temp_model = Model(cfg["model"])
    trainer_temp = Trainer(temp_model, ea_labeled, "config.yaml")
    groups = trainer_temp._assign_groups(ea_labeled)
    
    print(f"Groups created: {len(set(groups))} unique groups")
    for group_id in sorted(set(groups)):
        group_data = ea_labeled[groups == group_id]
        date_range = f"{group_data['Tweet Date'].min().date()} to {group_data['Tweet Date'].max().date()}"
        print(f"  Group {group_id}: {len(group_data)} tweets, dates {date_range}")
    
    # Quick training test
    print("\n⚙️  Setting up lightweight training...")
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    
    model = Model(model_cfg)
    trainer = Trainer(model, ea_labeled, "config.yaml")
    
    # Lightweight settings for quick test
    trainer.batch_size = 8
    trainer.epochs = 2
    
    print(f"🚀 Starting training on {len(ea_labeled)} tweets...")
    print("   (Expecting realistic validation metrics, NOT 99%+ accuracy)")
    
    trainer.train()
    
    print("\n📊 Training completed! Check training_metrics.json for results.")
    
    # Cross-validation prediction test
    if hasattr(trainer, "fold_states") and len(trainer.fold_states) > 0:
        print("\n🔮 Testing cross-validation predictions...")
        pred_df = cross_val_predict(trainer, ea_labeled)
        print(f"✅ Cross-val predictions: {len(pred_df)} rows")
        print(f"Prediction columns: {[col for col in pred_df.columns if 'Pred' in col]}")
        
        # Show sample predictions
        sample_preds = pred_df[['Tweet Date', 'Label', 'Pred_Label', 'Pred_Conf']].head(10)
        print("\n�� Sample predictions:")
        print(sample_preds.to_string(index=False))
    else:
        print("⚠️  No fold states found - cross-validation may have failed")

if __name__ == "__main__":
    test_temporal_grouping()
