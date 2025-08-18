#!/usr/bin/env python3
"""
Debug script to understand why the fixed MarketLabeler is still producing skewed results.
"""

import pandas as pd
import numpy as np
from market_labeler_fixed import MarketLabelerFixed

def debug_fixed_labeler():
    """Debug the fixed labeler step by step."""
    print("="*60)
    print("DEBUGGING FIXED MARKET LABELER")
    print("="*60)
    
    # Load a small sample of data
    df = pd.read_csv("data/combined_dataset_raw.csv")
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[df['Close'].notna()]
    
    # Take just a few days for detailed analysis
    start_date = '2015-01-01'
    end_date = '2015-01-10'
    df_sample = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    df_sample = df_sample.sort_values('date').reset_index(drop=True)
    
    print(f"Debug sample: {len(df_sample)} tweets from {start_date} to {end_date}")
    print(f"Unique dates: {df_sample['date'].dt.date.nunique()}")
    
    # Analyze daily price structure
    print("\n" + "="*40)
    print("DAILY PRICE ANALYSIS")
    print("="*40)
    
    daily_summary = df_sample.groupby(df_sample['date'].dt.date).agg({
        'Close': ['first', 'last', 'count'],
        'date': 'count'
    }).round(4)
    daily_summary.columns = ['Open_Price', 'Close_Price', 'Price_Records', 'Tweet_Count']
    daily_summary['Price_Change'] = daily_summary['Close_Price'] - daily_summary['Open_Price']
    daily_summary['Price_Change_Pct'] = (daily_summary['Price_Change'] / daily_summary['Open_Price'] * 100).round(4)
    
    print(daily_summary)
    
    # Check if all tweets on same day have same price
    print("\n" + "="*40)
    print("INTRA-DAY PRICE VARIANCE CHECK")
    print("="*40)
    
    for date in df_sample['date'].dt.date.unique()[:3]:
        day_data = df_sample[df_sample['date'].dt.date == date]
        print(f"\nDate: {date}")
        print(f"  Tweets: {len(day_data)}")
        print(f"  Price range: {day_data['Close'].min():.4f} - {day_data['Close'].max():.4f}")
        print(f"  Price variance: {day_data['Close'].var():.8f}")
        print(f"  Unique prices: {day_data['Close'].nunique()}")
    
    # Test the labeler with debug info
    print("\n" + "="*40)
    print("TESTING FIXED LABELER")
    print("="*40)
    
    labeler = MarketLabelerFixed("config.yaml")
    
    # Create a simple test case with known price movements
    test_data = pd.DataFrame({
        'date': pd.date_range('2015-01-01', periods=10, freq='D'),
        'Close': [100, 100, 105, 105, 95, 95, 110, 110, 90, 90],  # Clear movements
        'Tweet Content': ['tweet'] * 10,
        'Is_Event': [0] * 10
    })
    
    print("Test data (simple case):")
    print(test_data[['date', 'Close']])
    
    result = labeler.label_data(test_data)
    print("\nLabeling results:")
    print(result[['date', 'Close', 'Label', 'Upper Barrier', 'Lower Barrier']])
    print(f"\nLabel distribution: {result['Label'].value_counts().to_dict()}")
    
    # Test with actual data sample
    print("\n" + "="*40)
    print("TESTING WITH ACTUAL DATA SAMPLE")
    print("="*40)
    
    result_actual = labeler.label_data(df_sample)
    print(f"\nActual sample labeling:")
    print(f"Label distribution: {result_actual['Label'].value_counts().to_dict()}")
    print(f"Percentages: {(result_actual['Label'].value_counts() / len(result_actual) * 100).round(2).to_dict()}")
    
    # Show first few results with details
    print("\nFirst 10 labeled tweets:")
    cols = ['date', 'Close', 'Label', 'Upper Barrier', 'Lower Barrier', 'Vertical Barrier']
    print(result_actual[cols].head(10))


if __name__ == "__main__":
    debug_fixed_labeler() 