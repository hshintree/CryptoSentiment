#!/usr/bin/env python3
"""
Quality checker for the dataset loading and aggregation process.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class DatasetQualityChecker:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def check_raw_dataset(self, filename: str = "combined_dataset_raw.csv") -> dict:
        """Check the quality of the raw dataset."""
        print("=== Raw Dataset Quality Check ===")
        
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"❌ Raw dataset not found: {filepath}")
            return {}
        
        print(f"Loading raw dataset: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        
        results = {
            'file_size_mb': filepath.stat().st_size / 1024**2,
            'total_rows': len(df),
            'columns': list(df.columns),
            'date_range': (df['date'].min(), df['date'].max()),
            'unique_dates': df['date'].nunique(),
            'data_sources': df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {},
            'price_coverage': {},
            'follower_stats': {},
            'sample_data': {}
        }
        
        print(f"📊 Dataset Overview:")
        print(f"  File size: {results['file_size_mb']:.1f} MB")
        print(f"  Total rows: {results['total_rows']:,}")
        print(f"  Columns: {len(results['columns'])}")
        print(f"  Date range: {results['date_range'][0]} to {results['date_range'][1]}")
        print(f"  Unique dates: {results['unique_dates']:,}")
        
        # Check data sources
        if 'data_source' in df.columns:
            print(f"\n📈 Data Sources:")
            for source, count in results['data_sources'].items():
                print(f"  {source}: {count:,} tweets ({100 * count / results['total_rows']:.1f}%)")
        
        # Check price coverage
        if 'Close' in df.columns:
            price_coverage = df['Close'].notna().sum()
            price_percentage = 100 * price_coverage / results['total_rows']
            results['price_coverage'] = {
                'rows_with_price': price_coverage,
                'percentage': price_percentage,
                'price_range': (df['Close'].min(), df['Close'].max()) if price_coverage > 0 else (None, None)
            }
            print(f"\n💰 Price Coverage:")
            print(f"  Rows with price: {price_coverage:,} ({price_percentage:.2f}%)")
            if price_coverage > 0:
                print(f"  Price range: ${results['price_coverage']['price_range'][0]:.2f} - ${results['price_coverage']['price_range'][1]:.2f}")
        
        # Check user followers (for Kaggle tweets)
        if 'user_followers' in df.columns:
            kaggle_df = df[df['data_source'] == 'kaggle']
            if len(kaggle_df) > 0:
                follower_stats = kaggle_df['user_followers'].describe()
                results['follower_stats'] = {
                    'mean': follower_stats['mean'],
                    'median': follower_stats['50%'],
                    'min': follower_stats['min'],
                    'max': follower_stats['max'],
                    'std': follower_stats['std'],
                    'tweets_above_2000': (kaggle_df['user_followers'] >= 2000).sum(),
                    'tweets_below_2000': (kaggle_df['user_followers'] < 2000).sum()
                }
                print(f"\n👥 User Followers (Kaggle tweets):")
                print(f"  Mean: {results['follower_stats']['mean']:,.0f}")
                print(f"  Median: {results['follower_stats']['median']:,.0f}")
                print(f"  Range: {results['follower_stats']['min']:,.0f} - {results['follower_stats']['max']:,.0f}")
                print(f"  ≥2000 followers: {results['follower_stats']['tweets_above_2000']:,} tweets")
                print(f"  <2000 followers: {results['follower_stats']['tweets_below_2000']:,} tweets")
        
        # Sample data for inspection
        print(f"\n📝 Sample Data:")
        sample = df.head(3)
        for i, row in sample.iterrows():
            print(f"  Row {i}: {row['date']} | {row['Tweet Content'][:50]}... | Source: {row.get('data_source', 'N/A')}")
        
        results['sample_data'] = sample.to_dict('records')
        
        return results
    
    def check_aggregated_dataset(self, filename: str = "combined_dataset_aggregated.csv") -> dict:
        """Check the quality of the aggregated dataset."""
        print("\n=== Aggregated Dataset Quality Check ===")
        
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"❌ Aggregated dataset not found: {filepath}")
            return {}
        
        print(f"Loading aggregated dataset: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        
        results = {
            'file_size_mb': filepath.stat().st_size / 1024**2,
            'total_rows': len(df),
            'columns': list(df.columns),
            'date_range': (df['date'].min(), df['date'].max()),
            'unique_dates': df['date'].nunique(),
            'price_coverage': {},
            'expected_date_range': {},
            'aggregation_quality': {}
        }
        
        print(f"📊 Aggregated Dataset Overview:")
        print(f"  File size: {results['file_size_mb']:.1f} MB")
        print(f"  Total rows: {results['total_rows']:,}")
        print(f"  Columns: {len(results['columns'])}")
        print(f"  Date range: {results['date_range'][0]} to {results['date_range'][1]}")
        print(f"  Unique dates: {results['unique_dates']:,}")
        
        # Check if date range makes sense
        start_date = results['date_range'][0]
        end_date = results['date_range'][1]
        
        # Expected date range for Bitcoin data (2015-2025)
        expected_start = datetime(2015, 1, 1)
        expected_end = datetime(2025, 12, 31)
        
        results['expected_date_range'] = {
            'expected_start': expected_start,
            'expected_end': expected_end,
            'actual_start': start_date,
            'actual_end': end_date,
            'start_date_issue': start_date < expected_start,
            'end_date_issue': end_date > expected_end,
            'expected_days': (expected_end - expected_start).days,
            'actual_days': (end_date - start_date).days
        }
        
        print(f"\n📅 Date Range Analysis:")
        print(f"  Expected range: {expected_start} to {expected_end}")
        print(f"  Actual range: {start_date} to {end_date}")
        print(f"  Expected days: {results['expected_date_range']['expected_days']:,}")
        print(f"  Actual days: {results['expected_date_range']['actual_days']:,}")
        
        if results['expected_date_range']['start_date_issue']:
            print(f"  ⚠️  WARNING: Start date is too early!")
        if results['expected_date_range']['end_date_issue']:
            print(f"  ⚠️  WARNING: End date is too late!")
        
        # Check price coverage
        if 'Close' in df.columns:
            price_coverage = df['Close'].notna().sum()
            price_percentage = 100 * price_coverage / results['total_rows']
            results['price_coverage'] = {
                'rows_with_price': price_coverage,
                'percentage': price_percentage,
                'price_range': (df['Close'].min(), df['Close'].max()) if price_coverage > 0 else (None, None),
                'missing_prices': results['total_rows'] - price_coverage
            }
            print(f"\n💰 Price Coverage:")
            print(f"  Rows with price: {price_coverage:,} ({price_percentage:.2f}%)")
            print(f"  Missing prices: {results['price_coverage']['missing_prices']:,}")
            if price_coverage > 0:
                print(f"  Price range: ${results['price_coverage']['price_range'][0]:.2f} - ${results['price_coverage']['price_range'][1]:.2f}")
        
        # Check aggregation quality
        if 'Tweet Content' in df.columns:
            # Check for reasonable tweet content lengths
            tweet_lengths = df['Tweet Content'].str.len()
            results['aggregation_quality'] = {
                'mean_tweet_length': tweet_lengths.mean(),
                'median_tweet_length': tweet_lengths.median(),
                'max_tweet_length': tweet_lengths.max(),
                'min_tweet_length': tweet_lengths.min(),
                'empty_tweets': (tweet_lengths == 0).sum(),
                'very_short_tweets': (tweet_lengths < 10).sum(),
                'very_long_tweets': (tweet_lengths > 10000).sum()
            }
            
            print(f"\n📝 Tweet Content Quality:")
            print(f"  Mean length: {results['aggregation_quality']['mean_tweet_length']:.0f} chars")
            print(f"  Median length: {results['aggregation_quality']['median_tweet_length']:.0f} chars")
            print(f"  Length range: {results['aggregation_quality']['min_tweet_length']} - {results['aggregation_quality']['max_tweet_length']} chars")
            print(f"  Empty tweets: {results['aggregation_quality']['empty_tweets']}")
            print(f"  Very short tweets (<10 chars): {results['aggregation_quality']['very_short_tweets']}")
            print(f"  Very long tweets (>10k chars): {results['aggregation_quality']['very_long_tweets']}")
        
        # Sample aggregated data
        print(f"\n📝 Sample Aggregated Data:")
        sample = df.head(3)
        for i, row in sample.iterrows():
            tweet_preview = row['Tweet Content'][:100] + "..." if len(str(row['Tweet Content'])) > 100 else row['Tweet Content']
            print(f"  Row {i}: {row['date']} | {tweet_preview}")
        
        return results
    
    def diagnose_issues(self, raw_results: dict, agg_results: dict) -> dict:
        """Diagnose issues based on quality check results."""
        print("\n=== Issue Diagnosis ===")
        
        issues = []
        
        # Check for suspicious date ranges
        if raw_results and 'date_range' in raw_results:
            start_date = raw_results['date_range'][0]
            if start_date.year < 2010:
                issues.append({
                    'severity': 'CRITICAL',
                    'type': 'date_range',
                    'description': f'Raw dataset has suspicious start date: {start_date}',
                    'suggestion': 'Check date parsing in dataset loader'
                })
        
        if agg_results and 'expected_date_range' in agg_results:
            if agg_results['expected_date_range']['start_date_issue']:
                issues.append({
                    'severity': 'CRITICAL',
                    'type': 'date_range',
                    'description': f'Aggregated dataset has suspicious start date: {agg_results["expected_date_range"]["actual_start"]}',
                    'suggestion': 'Check date parsing and aggregation logic'
                })
        
        # Check for low price coverage
        if raw_results and 'price_coverage' in raw_results:
            if raw_results['price_coverage']['percentage'] < 50:
                issues.append({
                    'severity': 'HIGH',
                    'type': 'price_coverage',
                    'description': f'Low price coverage in raw dataset: {raw_results["price_coverage"]["percentage"]:.2f}%',
                    'suggestion': 'Check price data loading and merging logic'
                })
        
        if agg_results and 'price_coverage' in agg_results:
            if agg_results['price_coverage']['percentage'] < 50:
                issues.append({
                    'severity': 'HIGH',
                    'type': 'price_coverage',
                    'description': f'Low price coverage in aggregated dataset: {agg_results["price_coverage"]["percentage"]:.2f}%',
                    'suggestion': 'Check aggregation logic and price data preservation'
                })
        
        # Check for unreasonable dataset sizes
        if raw_results and raw_results['total_rows'] > 10000000:
            issues.append({
                'severity': 'MEDIUM',
                'type': 'dataset_size',
                'description': f'Very large raw dataset: {raw_results["total_rows"]:,} rows',
                'suggestion': 'Consider if this is expected or if there are duplicate rows'
            })
        
        if agg_results and agg_results['total_rows'] > 10000:
            issues.append({
                'severity': 'HIGH',
                'type': 'dataset_size',
                'description': f'Very large aggregated dataset: {agg_results["total_rows"]:,} rows (expected ~3,000-4,000)',
                'suggestion': 'Check aggregation logic - should have ~1 row per day'
            })
        
        # Print issues
        if issues:
            print("🚨 Issues Found:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. [{issue['severity']}] {issue['type'].upper()}: {issue['description']}")
                print(f"     💡 Suggestion: {issue['suggestion']}")
        else:
            print("✅ No major issues detected!")
        
        return {'issues': issues, 'issue_count': len(issues)}
    
    def run_full_check(self) -> dict:
        """Run a full quality check on both datasets."""
        print("🔍 Running Full Dataset Quality Check")
        print("=" * 50)
        
        # Check raw dataset
        raw_results = self.check_raw_dataset()
        
        # Check aggregated dataset
        agg_results = self.check_aggregated_dataset()
        
        # Diagnose issues
        diagnosis = self.diagnose_issues(raw_results, agg_results)
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 SUMMARY")
        print("=" * 50)
        
        if raw_results:
            print(f"Raw Dataset: {raw_results['total_rows']:,} rows, {raw_results['file_size_mb']:.1f} MB")
        if agg_results:
            print(f"Aggregated Dataset: {agg_results['total_rows']:,} rows, {agg_results['file_size_mb']:.1f} MB")
        
        print(f"Issues Found: {diagnosis['issue_count']}")
        
        return {
            'raw_results': raw_results,
            'agg_results': agg_results,
            'diagnosis': diagnosis
        }


def main():
    """Main function to run quality check."""
    checker = DatasetQualityChecker()
    results = checker.run_full_check()
    
    # Save results to file
    output_file = Path("data/quality_check_results.txt")
    with open(output_file, 'w') as f:
        f.write("Dataset Quality Check Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        if results['raw_results']:
            f.write("Raw Dataset:\n")
            f.write(f"  Rows: {results['raw_results']['total_rows']:,}\n")
            f.write(f"  Date range: {results['raw_results']['date_range']}\n")
            if 'price_coverage' in results['raw_results']:
                f.write(f"  Price coverage: {results['raw_results']['price_coverage']['percentage']:.2f}%\n")
        
        if results['agg_results']:
            f.write("\nAggregated Dataset:\n")
            f.write(f"  Rows: {results['agg_results']['total_rows']:,}\n")
            f.write(f"  Date range: {results['agg_results']['date_range']}\n")
            if 'price_coverage' in results['agg_results']:
                f.write(f"  Price coverage: {results['agg_results']['price_coverage']['percentage']:.2f}%\n")
        
        f.write(f"\nIssues: {results['diagnosis']['issue_count']}\n")
    
    print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main() 