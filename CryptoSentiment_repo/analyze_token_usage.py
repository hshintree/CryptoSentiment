#!/usr/bin/env python3
"""
Analyze token usage patterns in the dataset to understand BERT token limits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter


def analyze_token_usage():
    """Analyze token usage patterns in the dataset."""
    print("=== Token Usage Analysis ===")
    
    # Load the aggregated dataset
    data_dir = Path("data")
    aggregated_file = data_dir / "combined_dataset_aggregated.csv"
    
    if not aggregated_file.exists():
        print("❌ No aggregated dataset found. Please run the dataset loader first.")
        return
    
    print(f"Loading dataset: {aggregated_file}")
    df = pd.read_csv(aggregated_file, parse_dates=["date"])
    print(f"Dataset: {len(df)} rows")
    
    # Analyze tweet lengths
    print("\n=== Tweet Length Analysis ===")
    
    # Calculate various length metrics
    df['char_count'] = df['Tweet Content'].str.len()
    df['word_count'] = df['Tweet Content'].str.split().str.len()
    df['estimated_tokens'] = df['word_count'] * 1.3  # Rough BERT token estimate
    
    print(f"Character count statistics:")
    print(f"  Mean: {df['char_count'].mean():.1f}")
    print(f"  Median: {df['char_count'].median():.1f}")
    print(f"  Min: {df['char_count'].min()}")
    print(f"  Max: {df['char_count'].max()}")
    print(f"  95th percentile: {df['char_count'].quantile(0.95):.1f}")
    
    print(f"\nWord count statistics:")
    print(f"  Mean: {df['word_count'].mean():.1f}")
    print(f"  Median: {df['word_count'].median():.1f}")
    print(f"  Min: {df['word_count'].min()}")
    print(f"  Max: {df['word_count'].max()}")
    print(f"  95th percentile: {df['word_count'].quantile(0.95):.1f}")
    
    print(f"\nEstimated BERT tokens:")
    print(f"  Mean: {df['estimated_tokens'].mean():.1f}")
    print(f"  Median: {df['estimated_tokens'].median():.1f}")
    print(f"  Min: {df['estimated_tokens'].min():.1f}")
    print(f"  Max: {df['estimated_tokens'].max():.1f}")
    print(f"  95th percentile: {df['estimated_tokens'].quantile(0.95):.1f}")
    
    # Analyze by data source
    if 'data_source' in df.columns:
        print(f"\n=== Analysis by Data Source ===")
        for source in df['data_source'].unique():
            source_df = df[df['data_source'] == source]
            print(f"\n{source.upper()} tweets ({len(source_df)} rows):")
            print(f"  Mean tokens: {source_df['estimated_tokens'].mean():.1f}")
            print(f"  Median tokens: {source_df['estimated_tokens'].median():.1f}")
            print(f"  Max tokens: {source_df['estimated_tokens'].max():.1f}")
            print(f"  >400 tokens: {(source_df['estimated_tokens'] > 400).sum()} tweets")
    
    # Analyze daily aggregation impact
    print(f"\n=== Daily Aggregation Analysis ===")
    
    # Group by date and calculate total tokens per day
    daily_tokens = df.groupby('date')['estimated_tokens'].sum()
    
    print(f"Daily token usage statistics:")
    print(f"  Mean tokens per day: {daily_tokens.mean():.1f}")
    print(f"  Median tokens per day: {daily_tokens.median():.1f}")
    print(f"  Min tokens per day: {daily_tokens.min():.1f}")
    print(f"  Max tokens per day: {daily_tokens.max():.1f}")
    print(f"  95th percentile: {daily_tokens.quantile(0.95):.1f}")
    
    # Count days that would exceed BERT limits
    bert_limit = 512
    days_over_limit = (daily_tokens > bert_limit).sum()
    print(f"\nDays exceeding {bert_limit} tokens: {days_over_limit}/{len(daily_tokens)} ({100 * days_over_limit / len(daily_tokens):.1f}%)")
    
    # Show examples of very long days
    print(f"\nTop 5 days with most tokens:")
    top_days = daily_tokens.nlargest(5)
    for date, tokens in top_days.items():
        day_tweets = df[df['date'] == date]
        print(f"  {date}: {tokens:.1f} tokens ({len(day_tweets)} tweets)")
    
    # Analyze user followers impact (if available)
    if 'user_followers' in df.columns:
        print(f"\n=== User Followers Analysis ===")
        
        # Filter to Kaggle tweets only
        kaggle_df = df[df['data_source'] == 'kaggle'].copy()
        if len(kaggle_df) > 0:
            print(f"Kaggle tweets with follower data: {len(kaggle_df)}")
            
            # Analyze by follower ranges
            follower_ranges = [
                (0, 1000, "0-1K"),
                (1000, 5000, "1K-5K"),
                (5000, 10000, "5K-10K"),
                (10000, 50000, "10K-50K"),
                (50000, float('inf'), "50K+")
            ]
            
            for min_followers, max_followers, label in follower_ranges:
                if max_followers == float('inf'):
                    mask = kaggle_df['user_followers'] >= min_followers
                else:
                    mask = (kaggle_df['user_followers'] >= min_followers) & (kaggle_df['user_followers'] < max_followers)
                
                range_df = kaggle_df[mask]
                if len(range_df) > 0:
                    print(f"  {label} followers ({len(range_df)} tweets):")
                    print(f"    Mean tokens: {range_df['estimated_tokens'].mean():.1f}")
                    print(f"    Median tokens: {range_df['estimated_tokens'].median():.1f}")
                    print(f"    Max tokens: {range_df['estimated_tokens'].max():.1f}")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    
    # Calculate optimal follower threshold
    if 'user_followers' in df.columns:
        kaggle_df = df[df['data_source'] == 'kaggle'].copy()
        if len(kaggle_df) > 0:
            thresholds = [1000, 2000, 5000, 10000]
            print(f"Impact of different follower thresholds:")
            for threshold in thresholds:
                high_follower = kaggle_df[kaggle_df['user_followers'] >= threshold]
                remaining = len(high_follower)
                total = len(kaggle_df)
                print(f"  ≥{threshold} followers: {remaining}/{total} tweets ({100 * remaining / total:.1f}%)")
    
    # Token limit recommendations
    print(f"\nBERT Token Limit Recommendations:")
    print(f"  Current max daily tokens: {daily_tokens.max():.1f}")
    print(f"  Recommended daily limit: 512 tokens")
    print(f"  Days needing filtering: {days_over_limit}")
    
    if days_over_limit > 0:
        print(f"  Suggested strategies:")
        print(f"    1. Filter by user followers (≥2000 recommended)")
        print(f"    2. Truncate very long tweets (>400 tokens)")
        print(f"    3. Prioritize tweets by follower count and length")
    
    # Save analysis results
    analysis_file = data_dir / "token_analysis.csv"
    analysis_df = df[['date', 'Tweet Content', 'char_count', 'word_count', 'estimated_tokens', 'data_source']].copy()
    if 'user_followers' in df.columns:
        analysis_df['user_followers'] = df['user_followers']
    
    analysis_df.to_csv(analysis_file, index=False)
    print(f"\n✅ Analysis saved to: {analysis_file}")


def main():
    """Main function."""
    try:
        analyze_token_usage()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 