#!/usr/bin/env python3
"""
Fast aggregator test that works directly with the existing CSV file.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def fast_aggregate_test():
    """Test aggregation directly on the existing CSV file."""
    print("🚀 Fast Aggregator Test")
    print("=" * 50)
    
    # Load the existing raw dataset
    csv_file = Path("data/combined_dataset_raw.csv")
    if not csv_file.exists():
        print(f"❌ Raw dataset not found: {csv_file}")
        return
    
    print(f"Loading existing dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"📊 Dataset Overview:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    # Fix date parsing issue
    print(f"\n🔧 Fixing date parsing...")
    try:
        # Try parsing with mixed format to handle different date formats
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        print(f"  Date parsing successful")
    except Exception as e:
        print(f"  Date parsing failed: {e}")
        # Try alternative approach
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"  Date parsing with errors='coerce' successful")
    
    # Check for any failed date parsing
    failed_dates = df['date'].isna().sum()
    if failed_dates > 0:
        print(f"  ⚠️  {failed_dates} rows with failed date parsing")
        df = df.dropna(subset=['date'])
        print(f"  Removed {failed_dates} rows with invalid dates")
    
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique dates: {df['date'].nunique():,}")
    
    # Check the issue - are we getting too many unique dates?
    if df['date'].nunique() > 10000:
        print(f"\n🚨 ISSUE: Too many unique dates ({df['date'].nunique():,})")
        print(f"  Expected: ~3,000 days for 2015-2023")
        print(f"  This suggests tweets have different timestamps within the same day")
        
        # Show some examples of the same day with different timestamps
        sample_dates = df['date'].dt.date.value_counts().head(5)
        print(f"\n📅 Sample dates with multiple timestamps:")
        for date, count in sample_dates.items():
            print(f"  {date}: {count} timestamps")
            
            # Show a few timestamps for this date
            day_tweets = df[df['date'].dt.date == date]
            timestamps = day_tweets['date'].dt.time.unique()[:5]
            print(f"    Sample times: {timestamps}")
    
    # Test aggregation
    print(f"\n🔄 Testing Aggregation...")
    
    # Convert to date only for proper daily aggregation
    df['date_only'] = df['date'].dt.date
    print(f"  Unique dates after conversion: {df['date_only'].nunique():,}")
    
    # Simple aggregation test
    print(f"\n📈 Simple Aggregation Test:")
    
    # Group by date and count tweets
    daily_counts = df.groupby('date_only').size()
    print(f"  Days with tweets: {len(daily_counts):,}")
    print(f"  Average tweets per day: {daily_counts.mean():.1f}")
    print(f"  Max tweets per day: {daily_counts.max():,}")
    print(f"  Min tweets per day: {daily_counts.min():,}")
    
    # Show distribution
    print(f"\n📊 Tweet Distribution:")
    print(f"  Days with 1-10 tweets: {(daily_counts <= 10).sum():,}")
    print(f"  Days with 11-100 tweets: {((daily_counts > 10) & (daily_counts <= 100)).sum():,}")
    print(f"  Days with 101-1000 tweets: {((daily_counts > 100) & (daily_counts <= 1000)).sum():,}")
    print(f"  Days with >1000 tweets: {(daily_counts > 1000).sum():,}")
    
    # Test full aggregation
    print(f"\n🔧 Full Aggregation Test...")
    
    def agg_function(series):
        if series.name == 'Tweet Content':
            return ' [SEP] '.join(series.astype(str))
        elif series.name in ['Close', 'Volume']:
            return series.dropna().iloc[0] if not series.dropna().empty else series.iloc[0]
        else:
            return series.iloc[0]
    
    # Group by date and aggregate
    aggregated = df.groupby('date_only').agg(agg_function).reset_index()
    aggregated['date'] = pd.to_datetime(aggregated['date_only'])
    aggregated = aggregated.drop(columns=['date_only'])
    
    print(f"✅ Aggregation complete!")
    print(f"  Aggregated rows: {len(aggregated):,}")
    print(f"  Date range: {aggregated['date'].min()} to {aggregated['date'].max()}")
    
    # Check price coverage
    if 'Close' in aggregated.columns:
        price_coverage = aggregated['Close'].notna().sum()
        print(f"  Price coverage: {price_coverage:,}/{len(aggregated):,} ({100 * price_coverage / len(aggregated):.2f}%)")
    
    # Save test result
    test_file = Path("data/test_aggregated.csv")
    aggregated.to_csv(test_file, index=False)
    print(f"\n💾 Test result saved to: {test_file}")
    
    return aggregated


def main():
    """Main function."""
    try:
        result = fast_aggregate_test()
        if result is not None:
            print(f"\n🎉 Fast aggregation test successful!")
            print(f"Ready to proceed with full aggregation")
        else:
            print(f"\n❌ Fast aggregation test failed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 