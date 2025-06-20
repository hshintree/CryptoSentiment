#!/usr/bin/env python3
"""
Simple quality check for the datasets.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def check_dataset_quality():
    """Check the quality of both raw and aggregated datasets."""
    print("🔍 Dataset Quality Check")
    print("=" * 50)
    
    data_dir = Path("data")
    
    # Check raw dataset
    raw_file = data_dir / "combined_dataset_raw.csv"
    if raw_file.exists():
        print(f"\n📊 Raw Dataset: {raw_file}")
        df_raw = pd.read_csv(raw_file, parse_dates=["date"])
        
        print(f"  Rows: {len(df_raw):,}")
        print(f"  Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
        print(f"  Unique dates: {df_raw['date'].nunique():,}")
        
        # Check for suspicious dates
        suspicious_raw = df_raw[df_raw['date'].dt.year < 2010]
        if len(suspicious_raw) > 0:
            print(f"  ⚠️  SUSPICIOUS DATES: {len(suspicious_raw):,} rows with dates before 2010")
            print(f"     Sample dates: {suspicious_raw['date'].head(5).tolist()}")
        
        # Check price coverage
        if 'Close' in df_raw.columns:
            price_coverage = df_raw['Close'].notna().sum()
            print(f"  Price coverage: {price_coverage:,}/{len(df_raw):,} ({100 * price_coverage / len(df_raw):.2f}%)")
        
        # Check data sources
        if 'data_source' in df_raw.columns:
            sources = df_raw['data_source'].value_counts()
            print(f"  Data sources: {sources.to_dict()}")
    else:
        print(f"\n❌ Raw dataset not found: {raw_file}")
    
    # Check aggregated dataset
    agg_file = data_dir / "combined_dataset_aggregated.csv"
    if agg_file.exists():
        print(f"\n📊 Aggregated Dataset: {agg_file}")
        df_agg = pd.read_csv(agg_file, parse_dates=["date"])
        
        print(f"  Rows: {len(df_agg):,}")
        print(f"  Date range: {df_agg['date'].min()} to {df_agg['date'].max()}")
        print(f"  Unique dates: {df_agg['date'].nunique():,}")
        
        # Check for suspicious dates
        suspicious_agg = df_agg[df_agg['date'].dt.year < 2010]
        if len(suspicious_agg) > 0:
            print(f"  ⚠️  SUSPICIOUS DATES: {len(suspicious_agg):,} rows with dates before 2010")
            print(f"     Sample dates: {suspicious_agg['date'].head(5).tolist()}")
        
        # Check price coverage
        if 'Close' in df_agg.columns:
            price_coverage = df_agg['Close'].notna().sum()
            print(f"  Price coverage: {price_coverage:,}/{len(df_agg):,} ({100 * price_coverage / len(df_agg):.2f}%)")
        
        # Check if this makes sense (should be ~3000-4000 days for 2015-2025)
        expected_days = (datetime(2025, 12, 31) - datetime(2015, 1, 1)).days
        if len(df_agg) > expected_days * 2:
            print(f"  ⚠️  TOO MANY ROWS: Expected ~{expected_days:,} days, got {len(df_agg):,}")
        
    else:
        print(f"\n❌ Aggregated dataset not found: {agg_file}")
    
    print("\n" + "=" * 50)
    print("💡 Recommendations:")
    print("1. If you see suspicious dates (before 2010), check date parsing in the loader")
    print("2. If aggregated dataset has too many rows, check aggregation logic")
    print("3. If price coverage is low, check price data merging")
    print("4. Run the dataset loader with new debugging to see where issues occur")


if __name__ == "__main__":
    check_dataset_quality() 