#!/usr/bin/env python3
"""
Debug script to analyze dataset loading and market labeling issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from market_labeler import MarketLabeler
from dataset_loader import DatasetLoader


def analyze_dataset_and_labeling():
    """Comprehensive analysis of dataset loading and market labeling."""
    
    print("="*80)
    print("DATASET & MARKET LABELING ANALYSIS")
    print("="*80)
    
    # 1. Load raw dataset
    print("\n1. LOADING RAW DATASET")
    print("-" * 40)
    
    csv_path = "data/combined_dataset_raw.csv"
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        return
    
    # Load with robust date parsing
    df = pd.read_csv(csv_path)
    print(f"Raw dataset loaded: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Parse dates robustly
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    
    # Drop invalid dates
    before_date_filter = len(df)
    df = df.dropna(subset=['date'])
    print(f"After date filtering: {len(df)} rows (dropped {before_date_filter - len(df)})")
    
    # 2. Analyze data sources and years
    print("\n2. DATA SOURCE & YEAR ANALYSIS")
    print("-" * 40)
    
    df['year'] = df['date'].dt.year
    
    print("Data sources:")
    print(df['data_source'].value_counts())
    
    print("\nTweets per year:")
    year_counts = df['year'].value_counts().sort_index()
    print(year_counts)
    
    print("\nTweets per year by source:")
    source_year = df.groupby(['year', 'data_source']).size().unstack(fill_value=0)
    print(source_year)
    
    # 3. Analyze price data availability
    print("\n3. PRICE DATA ANALYSIS")
    print("-" * 40)
    
    print(f"Total rows: {len(df)}")
    print(f"Rows with Close price: {df['Close'].notna().sum()} ({100 * df['Close'].notna().mean():.1f}%)")
    print(f"Rows missing Close price: {df['Close'].isna().sum()} ({100 * df['Close'].isna().mean():.1f}%)")
    
    # Check price availability by year
    price_by_year = df.groupby('year')['Close'].agg(['count', lambda x: x.notna().sum(), lambda x: x.notna().mean()])
    price_by_year.columns = ['total', 'with_price', 'price_pct']
    print("\nPrice availability by year:")
    print(price_by_year)
    
    # 4. Filter to rows with valid price data for labeling
    print("\n4. FILTERING FOR MARKET LABELING")
    print("-" * 40)
    
    df_for_labeling = df[df['Close'].notna()].copy()
    print(f"Rows suitable for market labeling: {len(df_for_labeling)}")
    
    if len(df_for_labeling) == 0:
        print("ERROR: No rows with valid price data for labeling!")
        return
    
    # Sort by date for proper labeling
    df_for_labeling = df_for_labeling.sort_values('date').reset_index(drop=True)
    
    # 5. Apply market labeling
    print("\n5. APPLYING MARKET LABELING")
    print("-" * 40)
    
    labeler = MarketLabeler("config.yaml")
    print("Market labeler initialized")
    
    # Apply labeling
    df_labeled = labeler.label_data(df_for_labeling)
    print("Market labeling complete")
    
    # 6. Analyze labeling results
    print("\n6. MARKET LABELING RESULTS")
    print("-" * 40)
    
    label_counts = df_labeled['Label'].value_counts()
    print("Overall label distribution:")
    print(label_counts)
    print(f"Percentages: {(label_counts / len(df_labeled) * 100).round(2)}")
    
    # Analyze by year
    print("\nLabel distribution by year:")
    label_by_year = df_labeled.groupby('year')['Label'].value_counts().unstack(fill_value=0)
    print(label_by_year)
    
    # Analyze by source
    print("\nLabel distribution by data source:")
    label_by_source = df_labeled.groupby('data_source')['Label'].value_counts().unstack(fill_value=0)
    print(label_by_source)
    
    # 7. Analyze triple-barrier blocking
    print("\n7. TRIPLE-BARRIER ANALYSIS")
    print("-" * 40)
    
    # Look at barrier hit rates
    if 'Upper Barrier' in df_labeled.columns and 'Lower Barrier' in df_labeled.columns:
        df_labeled['price_change_pct'] = (df_labeled['Close'].shift(-1) - df_labeled['Close']) / df_labeled['Close'] * 100
        
        # Analyze what percentage of price movements would hit barriers
        upper_threshold = (df_labeled['Upper Barrier'] - df_labeled['Close']) / df_labeled['Close'] * 100
        lower_threshold = (df_labeled['Close'] - df_labeled['Lower Barrier']) / df_labeled['Close'] * 100
        
        print(f"Average upper barrier threshold: {upper_threshold.mean():.2f}%")
        print(f"Average lower barrier threshold: {lower_threshold.mean():.2f}%")
        
        # Check actual price movements vs barriers
        price_moves = df_labeled['price_change_pct'].abs()
        barrier_width = (upper_threshold + lower_threshold) / 2
        
        print(f"Average absolute price movement: {price_moves.mean():.2f}%")
        print(f"Average barrier width: {barrier_width.mean():.2f}%")
        print(f"Price moves > barrier width: {(price_moves > barrier_width).mean():.2%}")
    
    # 8. Analyze specific periods
    print("\n8. SPECIFIC PERIOD ANALYSIS")
    print("-" * 40)
    
    # 2020 analysis (Ea dataset)
    df_2020 = df_labeled[df_labeled['year'] == 2020]
    if len(df_2020) > 0:
        print(f"\n2020 data (Ea dataset): {len(df_2020)} tweets")
        print("2020 label distribution:")
        print(df_2020['Label'].value_counts())
        print(f"2020 percentages: {(df_2020['Label'].value_counts() / len(df_2020) * 100).round(2)}")
    
    # Event analysis
    if 'Is_Event' in df_labeled.columns:
        event_labels = df_labeled[df_labeled['Is_Event'] == 1]['Label'].value_counts()
        non_event_labels = df_labeled[df_labeled['Is_Event'] == 0]['Label'].value_counts()
        
        print(f"\nEvent tweets ({df_labeled['Is_Event'].sum()} total):")
        print(event_labels)
        
        print(f"\nNon-event tweets ({(df_labeled['Is_Event'] == 0).sum()} total):")
        print(non_event_labels)
    
    # 9. Diagnostic plots
    print("\n9. CREATING DIAGNOSTIC PLOTS")
    print("-" * 40)
    
    try:
        # Plot label distribution by year
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        label_counts.plot(kind='bar')
        plt.title('Overall Label Distribution')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 2)
        label_by_year.plot(kind='bar', stacked=True)
        plt.title('Label Distribution by Year')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 3)
        if len(label_by_source) > 0:
            label_by_source.plot(kind='bar')
            plt.title('Label Distribution by Source')
            plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 4)
        df_labeled['year'].hist(bins=20)
        plt.title('Tweet Distribution by Year')
        
        plt.subplot(2, 3, 5)
        if 'Close' in df_labeled.columns:
            df_labeled['Close'].hist(bins=50)
            plt.title('Bitcoin Price Distribution')
        
        plt.subplot(2, 3, 6)
        if 'price_change_pct' in df_labeled.columns:
            df_labeled['price_change_pct'].hist(bins=50, range=(-10, 10))
            plt.title('Daily Price Change % Distribution')
        
        plt.tight_layout()
        plt.savefig('labeling_analysis.png', dpi=150, bbox_inches='tight')
        print("Diagnostic plots saved to 'labeling_analysis.png'")
        
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # 10. Recommendations
    print("\n10. RECOMMENDATIONS")
    print("-" * 40)
    
    neutral_pct = label_counts.get('Neutral', 0) / len(df_labeled)
    bullish_pct = label_counts.get('Bullish', 0) / len(df_labeled)
    bearish_pct = label_counts.get('Bearish', 0) / len(df_labeled)
    
    print(f"Current distribution: {neutral_pct:.1%} Neutral, {bullish_pct:.1%} Bullish, {bearish_pct:.1%} Bearish")
    
    if neutral_pct > 0.9:
        print("\n⚠️  ISSUE: Extremely high Neutral percentage (>90%)")
        print("Possible causes:")
        print("1. Barrier thresholds too wide - price rarely hits barriers")
        print("2. Vertical barrier (time horizon) too short")
        print("3. Price data granularity issues (daily vs intraday)")
        print("4. Market labeling parameters need tuning")
        
        print("\nSuggested fixes:")
        print("1. Reduce barrier thresholds (fu_grid, fl_grid) in config.yaml")
        print("2. Increase vertical barrier horizon (vt_grid)")
        print("3. Check if using proper price data (Close prices)")
        print("4. Consider using intraday price data if available")
    
    if len(df_2020) < 50000:
        print(f"\n⚠️  WARNING: Only {len(df_2020)} tweets in 2020, target was ~60k")
        print("Check if 2020 data is properly loaded and filtered")
    
    return df_labeled


if __name__ == "__main__":
    result = analyze_dataset_and_labeling() 