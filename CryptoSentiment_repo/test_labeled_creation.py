#!/usr/bin/env python3
"""
Test script for labeled dataset creation.
"""

import pandas as pd
from pathlib import Path
from create_labeled_dataset import LabeledDatasetCreator


def test_labeled_creation():
    """Test the labeled dataset creation process."""
    print("=== Testing Labeled Dataset Creation ===")
    
    # Check if we have the required saved dataset
    data_dir = Path("data")
    aggregated_file = data_dir / "combined_dataset_aggregated.csv"
    
    if not aggregated_file.exists():
        print("❌ No aggregated dataset found. Please run the dataset loader first.")
        print("   Expected file: data/combined_dataset_aggregated.csv")
        return False
    
    print(f"✅ Found aggregated dataset: {aggregated_file}")
    
    # Load a small sample to test
    print("\nLoading sample from aggregated dataset...")
    df_sample = pd.read_csv(aggregated_file, parse_dates=["date"])
    print(f"Sample size: {len(df_sample)} rows")
    print(f"Date range: {df_sample['date'].min()} to {df_sample['date'].max()}")
    print(f"Columns: {list(df_sample.columns)}")
    
    # Test the creator with a small sample
    print("\n=== Testing LabeledDatasetCreator ===")
    creator = LabeledDatasetCreator()
    
    try:
        # Test with just the first 100 rows
        df_test = df_sample.head(100).copy()
        
        # Save test sample
        test_file = data_dir / "test_sample.csv"
        df_test.to_csv(test_file, index=False)
        print(f"Saved test sample: {test_file}")
        
        # Test preprocessing
        print("\nTesting preprocessing...")
        df_preprocessed = creator.preprocessor.preprocess(df_test)
        print(f"Preprocessing complete. New columns: {[col for col in df_preprocessed.columns if col not in df_test.columns]}")
        
        # Test market labeling
        print("\nTesting market labeling...")
        df_labeled = creator.market_labeler.label_data(df_preprocessed)
        print(f"Market labeling complete. Labels: {df_labeled['Label'].value_counts().to_dict()}")
        
        # Test filtering
        print("\nTesting BERT filtering...")
        df_filtered = creator._filter_tweets_for_bert(df_test)
        print(f"Filtering complete. Rows: {len(df_test)} -> {len(df_filtered)}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_labeled_creation()
    
    if success:
        print("\n🎉 Ready to create full labeled dataset!")
        print("Run: python create_labeled_dataset.py")
    else:
        print("\n⚠️  Please fix issues before creating full dataset")


if __name__ == "__main__":
    main() 