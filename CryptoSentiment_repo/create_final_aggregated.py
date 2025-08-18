#!/usr/bin/env python3
"""
Create the final aggregated dataset from the existing raw dataset.
"""

import pandas as pd
from pathlib import Path
from dataset_loader import DatasetLoader


def create_final_aggregated():
    """Create the final aggregated dataset."""
    print("🚀 Creating Final Aggregated Dataset")
    print("=" * 50)
    
    # Initialize the dataset loader
    loader = DatasetLoader()
    
    try:
        # Load and aggregate the saved raw dataset
        print("Loading saved raw dataset and aggregating...")
        df_agg = loader.aggregate_saved_dataset()
        
        print(f"\n✅ SUCCESS!")
        print(f"Final aggregated dataset created:")
        print(f"  Rows: {len(df_agg):,}")
        print(f"  Date range: {df_agg['date'].min()} to {df_agg['date'].max()}")
        print(f"  Price coverage: {df_agg['Close'].notna().sum():,}/{len(df_agg):,} ({100 * df_agg['Close'].notna().mean():.2f}%)")
        print(f"  File saved: data/combined_dataset_aggregated.csv")
        
        # Show sample data
        print(f"\n📝 Sample aggregated data:")
        for i, row in df_agg.head(3).iterrows():
            tweet_preview = row['Tweet Content'][:100] + "..." if len(str(row['Tweet Content'])) > 100 else row['Tweet Content']
            print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row['Close']:.2f} - {tweet_preview}")
        
        print(f"\n🎉 Ready for preprocessing and labeling!")
        
    except Exception as e:
        print(f"\n❌ Error creating aggregated dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_final_aggregated() 