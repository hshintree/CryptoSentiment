import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from eval_loader import EvalLoader

def test_ea_dataset_properties():
    """Test Ea (2020 in-sample) dataset properties."""
    print("\n=== Testing Ea Dataset Properties ===")
    
    # Load datasets
    loader = EvalLoader("config.yaml")
    datasets = loader.create_eval_datasets()
    ea_df = datasets['ea_full']
    
    # Basic dataset checks
    print(f"Total samples in Ea: {len(ea_df):,}")
    
    # Verify year is 2020
    years = ea_df['date'].dt.year.unique()
    print(f"Years present in Ea: {years}")
    assert all(year == 2020 for year in years), "Ea contains non-2020 data!"
    
    # Check label distribution
    label_dist = ea_df['Label'].value_counts(normalize=True)
    print("\nLabel distribution in Ea:")
    for label, pct in label_dist.items():
        print(f"{label}: {pct:.1%}")
    
    # Check if distribution is roughly balanced (within 5% of 1/3)
    assert all(abs(pct - 1/3) < 0.05 for pct in label_dist), "Labels are not balanced!"
    
    # Display feature information
    print("\nFeatures available in Ea:")
    for col in ea_df.columns:
        non_null = ea_df[col].count()
        dtype = ea_df[col].dtype
        print(f"{col:15} | {dtype:12} | {non_null:,} non-null values")

def test_eb_dataset_properties():
    """Test Eb (event-focused) dataset properties."""
    print("\n=== Testing Eb Dataset Properties ===")
    
    # Load datasets
    loader = EvalLoader("config.yaml")
    datasets = loader.create_eval_datasets()
    eb_df = datasets['eb_full']
    
    # Basic dataset checks
    print(f"Total samples in Eb: {len(eb_df):,}")
    
    # Verify year is NOT 2020
    years = sorted(eb_df['date'].dt.year.unique())
    print(f"Years present in Eb: {years}")
    assert 2020 not in years, "Eb contains 2020 data!"
    
    # Check event coverage
    event_tweets = eb_df[eb_df['Is_Event'] == 1]
    print(f"\nEvent coverage in Eb:")
    print(f"Event tweets: {len(event_tweets):,} ({len(event_tweets)/len(eb_df):.1%})")
    
    # Check volatility through price changes
    eb_df['daily_returns'] = eb_df.groupby(eb_df['date'].dt.date)['Close'].transform(
        lambda x: x.pct_change()
    )
    volatility = eb_df.groupby(eb_df['date'].dt.date)['daily_returns'].std()
    print(f"Average daily volatility: {volatility.mean():.2%}")
    
    # Display feature information
    print("\nFeatures available in Eb:")
    for col in eb_df.columns:
        non_null = eb_df[col].count()
        dtype = eb_df[col].dtype
        print(f"{col:15} | {dtype:12} | {non_null:,} non-null values")

def test_fold_integrity():
    """Test the integrity of time-series folds."""
    print("\n=== Testing Fold Integrity ===")
    
    loader = EvalLoader("config.yaml")
    datasets = loader.create_eval_datasets()
    
    # Test Ea folds
    print("\nEa Folds:")
    _check_fold_integrity(datasets['ea_train_folds'], datasets['ea_test_folds'], "Ea")
    
    # Test Eb folds
    print("\nEb Folds:")
    _check_fold_integrity(datasets['eb_train_folds'], datasets['eb_test_folds'], "Eb")

def _check_fold_integrity(train_folds, test_folds, dataset_name):
    """Helper to check fold properties."""
    for fold_idx, (train, test) in enumerate(zip(train_folds, test_folds)):
        # Check temporal separation
        train_max_date = train['date'].max()
        test_min_date = test['date'].min()
        is_sequential = train_max_date <= test_min_date
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"Train period: {train['date'].min().date()} to {train['date'].max().date()}")
        print(f"Test period:  {test['date'].min().date()} to {test['date'].max().date()}")
        print(f"Temporal separation maintained: {is_sequential}")
        
        # Check label distribution
        train_dist = train['Label'].value_counts(normalize=True)
        test_dist = test['Label'].value_counts(normalize=True)
        print("\nLabel distribution (train):")
        for label, pct in train_dist.items():
            print(f"{label}: {pct:.1%}")

if __name__ == "__main__":
    # Run all tests
    test_ea_dataset_properties()
    test_eb_dataset_properties()
    test_fold_integrity()