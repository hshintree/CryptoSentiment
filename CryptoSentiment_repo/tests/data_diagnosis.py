#!/usr/bin/env python3
"""
Data Diagnosis Script - Find potential data leakage and quality issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from collections import Counter
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from gpu_scripts.preprocessor import Preprocessor
from gpu_scripts.market_labeler_ewma import MarketLabelerTBL

def diagnose_dataset():
    print("🔍 DATASET DIAGNOSIS")
    print("=" * 50)
    
    # Load the exact same data as paper_tests.py
    print("📝 Loading and preprocessing data...")
    
    
    # Load raw data
    ea = pd.read_csv("data/#2val.csv", parse_dates=["date"])
    ea = ea.rename(columns={"date": "Tweet Date"})
    
    # Preprocess with new fit_transform API
    pp = Preprocessor("config.yaml")
    ea_pp = pp.fit_transform(ea)
    
    # Label with new TBL API
    ml = MarketLabelerTBL("config.yaml")
    ea_lbl = ml.fit_and_label(ea_pp)
    
    print(f"📊 Dataset Shape: {ea_lbl.shape}")
    print(f"📅 Date Range: {ea_lbl['Tweet Date'].min()} to {ea_lbl['Tweet Date'].max()}")
    
    # === 1. TEMPORAL ANALYSIS ===
    print("\n🕒 TEMPORAL ANALYSIS")
    print("-" * 30)
    
    # Check date distribution
    ea_lbl['date_only'] = ea_lbl['Tweet Date'].dt.date
    daily_counts = ea_lbl.groupby('date_only').size()
    
    print(f"📈 Unique dates: {len(daily_counts)}")
    print(f"📈 Total tweets: {len(ea_lbl)}")
    print(f"📈 Avg tweets per day: {daily_counts.mean():.1f}")
    print(f"📈 Max tweets per day: {daily_counts.max()}")
    print(f"📈 Min tweets per day: {daily_counts.min()}")
    
    # Check for temporal clustering
    print(f"\n📅 Date Distribution:")
    print(f"  Q1 (25%): {daily_counts.quantile(0.25):.1f} tweets/day")
    print(f"  Median:   {daily_counts.median():.1f} tweets/day") 
    print(f"  Q3 (75%): {daily_counts.quantile(0.75):.1f} tweets/day")
    
    # Days with many tweets (potential data quality issues)
    high_volume_days = daily_counts[daily_counts > daily_counts.quantile(0.95)]
    print(f"\n⚠️  High-volume days (top 5%):")
    print(high_volume_days.head(10))
    
    # === 2. CROSS-VALIDATION TEMPORAL LEAKAGE ===
    print("\n🔄 CV TEMPORAL LEAKAGE ANALYSIS")
    print("-" * 40)
    
    # Use proper TimeSeriesSplit like the updated trainer
    from sklearn.model_selection import TimeSeriesSplit
    
    # Create unique dates table (same as trainer logic)
    unique_dates = (
        ea_lbl.assign(__day=ea_lbl["Tweet Date"].dt.normalize())
            .drop_duplicates("__day")
            .sort_values("__day")
            .reset_index()
            .rename(columns={"__day": "Day"})
    )
    
    # Configure splits (same as trainer)
    gap_days = 15
    total_days = len(unique_dates)
    test_days = max(21, total_days // 5)
    max_splits = (total_days - gap_days) // test_days
    n_splits = min(5, max_splits)
    
    if n_splits < 2:
        print(f"⚠️  Too few days ({total_days}) for proper temporal CV")
        return
    
    # Day-to-rows mapping (same as trainer)
    day_series = ea_lbl["Tweet Date"].dt.normalize()
    day_to_rows = (
        ea_lbl.assign(__day=day_series)
            .groupby("__day", sort=False)["__day"]
            .apply(lambda g: g.index.values)
            .to_dict()
    )
    
    def expand(day_idx):
        """Convert day indices to tweet row indices."""
        days = unique_dates.loc[list(day_idx), "Day"]
        rows = []
        for d in days:
            rows.extend(day_to_rows[d])
        return np.asarray(rows, dtype=int)
    
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_days, gap=gap_days)
    print(f"📊 Using TimeSeriesSplit: {n_splits} splits, {test_days} test days, {gap_days} gap days")
    
    for fold, (day_tr, day_va) in enumerate(tss.split(unique_dates)):
        tr_idx = expand(day_tr)
        va_idx = expand(day_va)
        
        tr_dates = set(ea_lbl.iloc[tr_idx]['date_only'])
        va_dates = set(ea_lbl.iloc[va_idx]['date_only'])
        
        overlapping_dates = tr_dates & va_dates
        
        print(f"\n📁 Fold {fold + 1}:")
        print(f"  Train tweets: {len(tr_idx):,} | Val tweets: {len(va_idx):,}")
        print(f"  Train dates: {len(tr_dates)} unique days")
        print(f"  Val dates:   {len(va_dates)} unique days")
        print(f"  ✅ OVERLAPPING DATES: {len(overlapping_dates)} (should be 0)")
        
        if overlapping_dates:
            print(f"     🚨 Examples: {list(overlapping_dates)[:5]}")
            
        # Check temporal separation
        min_train_date = min(tr_dates) if tr_dates else None
        max_train_date = max(tr_dates) if tr_dates else None
        min_val_date = min(va_dates) if va_dates else None
        max_val_date = max(va_dates) if va_dates else None
        
        if all([min_train_date, max_train_date, min_val_date, max_val_date]):
            print(f"  Train range: {min_train_date} to {max_train_date}")
            print(f"  Val range:   {min_val_date} to {max_val_date}")
            
            # Check gap
            gap_actual = (min_val_date - max_train_date).days
            print(f"  ✅ Temporal gap: {gap_actual} days (target: {gap_days})")
        
        # Analyze label distributions for this fold
        train_labels = ea_lbl.iloc[tr_idx]['Label'].value_counts()
        val_labels = ea_lbl.iloc[va_idx]['Label'].value_counts()
        
        print(f"  📊 LABEL DISTRIBUTIONS:")
        print(f"    Training Set:")
        for label in ['Bearish', 'Neutral', 'Bullish']:
            count = train_labels.get(label, 0)
            pct = 100 * count / len(tr_idx) if len(tr_idx) > 0 else 0
            print(f"      {label}: {count:,} ({pct:.1f}%)")
        
        print(f"    Validation Set:")
        for label in ['Bearish', 'Neutral', 'Bullish']:
            count = val_labels.get(label, 0)
            pct = 100 * count / len(va_idx) if len(va_idx) > 0 else 0
            print(f"      {label}: {count:,} ({pct:.1f}%)")
        
        # Calculate label distribution differences between train/val
        print(f"    Distribution Differences (Train - Val):")
        for label in ['Bearish', 'Neutral', 'Bullish']:
            train_pct = 100 * train_labels.get(label, 0) / len(tr_idx) if len(tr_idx) > 0 else 0
            val_pct = 100 * val_labels.get(label, 0) / len(va_idx) if len(va_idx) > 0 else 0
            diff = train_pct - val_pct
            print(f"      {label}: {diff:+.1f}% {'⚠️' if abs(diff) > 10 else '✅'}")
        
        # Check for severe label imbalance
        if len(tr_idx) > 0:
            train_min_pct = min(100 * train_labels.get(label, 0) / len(tr_idx) for label in ['Bearish', 'Neutral', 'Bullish'])
            if train_min_pct < 5:
                print(f"      🚨 Severe train imbalance: min class {train_min_pct:.1f}%")
        
        if len(va_idx) > 0:
            val_min_pct = min(100 * val_labels.get(label, 0) / len(va_idx) for label in ['Bearish', 'Neutral', 'Bullish'])
            if val_min_pct < 5:
                print(f"      🚨 Severe val imbalance: min class {val_min_pct:.1f}%")
    
    # === CV LABEL DISTRIBUTION SUMMARY ===
    print(f"\n📈 CV LABEL DISTRIBUTION SUMMARY")
    print("-" * 35)
    
    # Collect all fold statistics
    fold_stats = []
    tss_summary = TimeSeriesSplit(n_splits=n_splits, test_size=test_days, gap=gap_days)
    for fold, (day_tr, day_va) in enumerate(tss_summary.split(unique_dates)):
        tr_idx = expand(day_tr)
        va_idx = expand(day_va)
        
        train_labels = ea_lbl.iloc[tr_idx]['Label'].value_counts()
        val_labels = ea_lbl.iloc[va_idx]['Label'].value_counts()
        
        fold_stat = {
            'fold': fold + 1,
            'train_bearish': 100 * train_labels.get('Bearish', 0) / len(tr_idx) if len(tr_idx) > 0 else 0,
            'train_neutral': 100 * train_labels.get('Neutral', 0) / len(tr_idx) if len(tr_idx) > 0 else 0,
            'train_bullish': 100 * train_labels.get('Bullish', 0) / len(tr_idx) if len(tr_idx) > 0 else 0,
            'val_bearish': 100 * val_labels.get('Bearish', 0) / len(va_idx) if len(va_idx) > 0 else 0,
            'val_neutral': 100 * val_labels.get('Neutral', 0) / len(va_idx) if len(va_idx) > 0 else 0,
            'val_bullish': 100 * val_labels.get('Bullish', 0) / len(va_idx) if len(va_idx) > 0 else 0,
        }
        fold_stats.append(fold_stat)
    
    # Display summary table
    print(f"{'Fold':<4} {'Train%':<20} {'Val%':<20} {'Differences':<20}")
    print(f"{'':4} {'Bear|Neut|Bull':<20} {'Bear|Neut|Bull':<20} {'Bear|Neut|Bull':<20}")
    print("-" * 65)
    
    for stat in fold_stats:
        train_str = f"{stat['train_bearish']:.1f}|{stat['train_neutral']:.1f}|{stat['train_bullish']:.1f}"
        val_str = f"{stat['val_bearish']:.1f}|{stat['val_neutral']:.1f}|{stat['val_bullish']:.1f}"
        diff_bear = stat['train_bearish'] - stat['val_bearish']
        diff_neut = stat['train_neutral'] - stat['val_neutral']
        diff_bull = stat['train_bullish'] - stat['val_bullish']
        diff_str = f"{diff_bear:+.1f}|{diff_neut:+.1f}|{diff_bull:+.1f}"
        print(f"{stat['fold']:<4} {train_str:<20} {val_str:<20} {diff_str:<20}")
    
    # Calculate cross-fold statistics
    if fold_stats:
        avg_train_bear = np.mean([s['train_bearish'] for s in fold_stats])
        avg_train_neut = np.mean([s['train_neutral'] for s in fold_stats])
        avg_train_bull = np.mean([s['train_bullish'] for s in fold_stats])
        avg_val_bear = np.mean([s['val_bearish'] for s in fold_stats])
        avg_val_neut = np.mean([s['val_neutral'] for s in fold_stats])
        avg_val_bull = np.mean([s['val_bullish'] for s in fold_stats])
        
        print("-" * 65)
        avg_train_str = f"{avg_train_bear:.1f}|{avg_train_neut:.1f}|{avg_train_bull:.1f}"
        avg_val_str = f"{avg_val_bear:.1f}|{avg_val_neut:.1f}|{avg_val_bull:.1f}"
        avg_diff_str = f"{avg_train_bear-avg_val_bear:+.1f}|{avg_train_neut-avg_val_neut:+.1f}|{avg_train_bull-avg_val_bull:+.1f}"
        print(f"{'Avg':<4} {avg_train_str:<20} {avg_val_str:<20} {avg_diff_str:<20}")
        
        # Check for concerning patterns
        max_diff = max(abs(avg_train_bear-avg_val_bear), abs(avg_train_neut-avg_val_neut), abs(avg_train_bull-avg_val_bull))
        if max_diff > 5:
            print(f"⚠️  Large average distribution difference: {max_diff:.1f}%")
        else:
            print(f"✅ Reasonable distribution consistency across folds")
     
     # === 3. LABEL ANALYSIS ===
    print("\n🏷️  LABEL ANALYSIS")
    print("-" * 20)
    
    label_dist = ea_lbl['Label'].value_counts()
    print(f"📊 Label Distribution:")
    for label, count in label_dist.items():
        print(f"  {label}: {count:,} ({100*count/len(ea_lbl):.1f}%)")
    
    # Check for temporal label patterns
    daily_labels = ea_lbl.groupby('date_only')['Label'].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
    daily_label_dist = daily_labels.value_counts()
    print(f"\n📊 Daily Majority Label Distribution:")
    for label, count in daily_label_dist.items():
        print(f"  {label}: {count:,} days ({100*count/len(daily_label_dist):.1f}%)")
    
    # Check for label runs/patterns
    label_changes = (daily_labels != daily_labels.shift()).sum()
    print(f"\n🔄 Label Changes: {label_changes} changes over {len(daily_labels)} days")
    print(f"   Average run length: {len(daily_labels) / label_changes:.1f} days")
    
    # === 4. DUPLICATE ANALYSIS ===
    print("\n🔍 DUPLICATE ANALYSIS")
    print("-" * 25)
    
    # Check for exact duplicates
    tweet_duplicates = ea_lbl['Tweet Content'].duplicated().sum()
    print(f"📄 Exact tweet duplicates: {tweet_duplicates}")
    
    # Check for very similar tweets (first 50 chars)
    ea_lbl['tweet_prefix'] = ea_lbl['Tweet Content'].str[:50]
    similar_tweets = ea_lbl['tweet_prefix'].duplicated().sum()
    print(f"📄 Similar tweets (same first 50 chars): {similar_tweets}")
    
    # Check for same tweets on same day
    same_day_same_tweet = ea_lbl.groupby(['date_only', 'Tweet Content']).size()
    repeated_tweets = (same_day_same_tweet > 1).sum()
    print(f"📄 Repeated tweets on same day: {repeated_tweets}")
    
    # === 5. FEATURE ANALYSIS ===
    print("\n📈 TECHNICAL INDICATOR ANALYSIS")
    print("-" * 35)
    
    # Check if technical indicators are too predictive
    print(f"RSI range: [{ea_lbl['RSI'].min():.2f}, {ea_lbl['RSI'].max():.2f}]")
    print(f"ROC range: [{ea_lbl['ROC'].min():.4f}, {ea_lbl['ROC'].max():.4f}]")
    print(f"Close range: [{ea_lbl['Close'].min():.2f}, {ea_lbl['Close'].max():.2f}]")
    
    # Check correlation between technical indicators and labels
    label_encoded = ea_lbl['Label'].map({'Bearish': 0, 'Neutral': 1, 'Bullish': 2})
    rsi_corr = label_encoded.corr(ea_lbl['RSI'])
    roc_corr = label_encoded.corr(ea_lbl['ROC'])
    
    print(f"\n🔗 Correlations with labels:")
    print(f"  RSI-Label correlation: {rsi_corr:.4f}")
    print(f"  ROC-Label correlation: {roc_corr:.4f}")
    
    # Check Previous Label correlation and distribution
    if 'Previous Label' in ea_lbl.columns:
        prev_label_encoded = ea_lbl['Previous Label'].map({'Bearish': 0, 'Neutral': 1, 'Bullish': 2})
        prev_corr = label_encoded.corr(prev_label_encoded)
        print(f"  Previous-Current Label correlation: {prev_corr:.4f}")
        
        # Check Previous Label distribution
        prev_dist = ea_lbl['Previous Label'].value_counts()
        print(f"\n📊 Previous Label Distribution:")
        for label, count in prev_dist.items():
            print(f"  {label}: {count:,} ({100*count/len(ea_lbl):.1f}%)")
    else:
        print(f"  ⚠️  No 'Previous Label' column found")
    
    if abs(rsi_corr) > 0.7 or abs(roc_corr) > 0.7:
        print("  🚨 WARNING: Very high correlation suggests potential leakage!")
    
    # === 6. SUMMARY & RECOMMENDATIONS ===
    print("\n🎯 DIAGNOSIS SUMMARY")
    print("=" * 25)
    
    issues_found = []
    
    # Check if we found any temporal leakage in the CV analysis
    temporal_leakage_found = False
    try:
        tss_check = TimeSeriesSplit(n_splits=n_splits, test_size=test_days, gap=gap_days)
        for fold, (day_tr, day_va) in enumerate(tss_check.split(unique_dates)):
            tr_idx_check = expand(day_tr)
            va_idx_check = expand(day_va)
            tr_dates_check = set(ea_lbl.iloc[tr_idx_check]['date_only'])
            va_dates_check = set(ea_lbl.iloc[va_idx_check]['date_only'])
            if len(tr_dates_check & va_dates_check) > 0:
                temporal_leakage_found = True
                break
    except:
        temporal_leakage_found = True  # Assume leakage if CV setup failed
    
    if temporal_leakage_found:
        issues_found.append("❌ TEMPORAL LEAKAGE: Same dates in train/val splits")
    
    if label_changes < len(daily_labels) / 10:
        issues_found.append("❌ LABEL PATTERNS: Very few label changes suggest artificial patterns")
    
    if abs(rsi_corr) > 0.5 or abs(roc_corr) > 0.5:
        issues_found.append("❌ FEATURE LEAKAGE: Technical indicators too predictive")
    
    if tweet_duplicates > len(ea_lbl) * 0.01:
        issues_found.append("❌ DATA QUALITY: Many duplicate tweets")
    
    # Check for label distribution issues across folds
    try:
        if 'fold_stats' in locals() and fold_stats:
            max_diff = max(abs(avg_train_bear-avg_val_bear), abs(avg_train_neut-avg_val_neut), abs(avg_train_bull-avg_val_bull))
            if max_diff > 10:
                issues_found.append("❌ LABEL DISTRIBUTION: Large train/val differences across folds")
            
            # Check for severe class imbalance in any fold
            min_class_pct = min([min(s['train_bearish'], s['train_neutral'], s['train_bullish'], 
                                    s['val_bearish'], s['val_neutral'], s['val_bullish']) for s in fold_stats])
            if min_class_pct < 5:
                issues_found.append("❌ CLASS IMBALANCE: Severe imbalance in some folds")
    except:
        pass
        
    if issues_found:
        print("🚨 CRITICAL ISSUES FOUND:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("✅ No obvious issues detected")
    
    print(f"\n📋 Next steps:")
    if temporal_leakage_found:
        print(f"  1. ❌ Fix temporal leakage - TimeSeriesSplit should prevent this")
    else:
        print(f"  1. ✅ Temporal CV looks good with TimeSeriesSplit")
    print(f"  2. Monitor label creation process for realistic distributions")
    print(f"  3. Verify preprocessor fit/transform prevents scaling leakage")
    print(f"  4. Check that MarketLabelerTBL prevents future information leakage")
    print(f"  5. Validate that Previous Label uses only past market data")

if __name__ == "__main__":
    diagnose_dataset() 