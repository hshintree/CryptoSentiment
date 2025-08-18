#!/usr/bin/env python3
"""
Quick test script for improved labeling with corrected volatility multipliers.
Loads the already-saved dataset to avoid waiting for data loading.
"""

import pandas as pd
from market_labeler_ewma import MarketLabelerTBL

def test_improved_labeling():
    print('🔍 TESTING IMPROVED LABELING WITH CORRECTED VOLATILITY MULTIPLIERS...')
    print('📁 Loading already-saved dataset to skip data loading time...')
    
    # Load the already-saved raw dataset
    try:
        df = pd.read_csv('data/combined_dataset_raw.csv')
        # Ensure date column is datetime with mixed format handling
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        print(f'✅ Loaded RAW dataset: {len(df):,} rows')
        
        # Check for any failed date conversions
        null_dates = df['date'].isna().sum()
        if null_dates > 0:
            print(f'⚠️  Warning: {null_dates} rows had unparseable dates')
            
    except FileNotFoundError:
        print('❌ combined_dataset_raw.csv not found! Run the full data loading first.')
        return
    
    # Apply labeling with NEW config (corrected volatility multipliers)
    print('🔬 Applying TBL labeling with corrected volatility multipliers...')
    labeler = MarketLabelerTBL('config.yaml')  # Using single config file
    df_labeled = labeler.label_data(df)
    print(f'✅ After labeling: {len(df_labeled):,} rows')

    # Check 2019-2020 data specifically (EA dataset)
    raw_ea = df_labeled[df_labeled['date'].dt.year.isin([2019, 2020])]
    print(f'\n📊 RAW EA DATA ANALYSIS (2019-2020 INDIVIDUAL TWEETS):')
    print(f'Total raw EA tweets: {len(raw_ea):,}')

    if len(raw_ea) > 0:
        label_counts = raw_ea['Label'].value_counts()
        print(f'Raw EA (2019-2020) label counts:')
        for label, count in label_counts.items():
            pct = count / len(raw_ea) * 100
            print(f'  {label}: {count:,} ({pct:.1f}%)')
        
        # Check if we have enough for balanced sampling
        min_class = label_counts.min()
        target_per_class = 30000  # ~30k per class for 90k total
        print(f'\nMinimum class size: {min_class:,}')
        print(f'Target per class: {target_per_class:,}')
        
        if min_class >= target_per_class:
            print('✅ Sufficient data for balanced 90K sampling (~30K per class)')
        else:
            shortage = target_per_class - min_class
            print(f'❌ Insufficient data - need {shortage:,} more tweets for smallest class')
            
            # Check if we can do smaller balanced sampling
            max_balanced = min_class * 3  # 3 classes
            print(f'💡 Could do balanced sampling with {max_balanced:,} total tweets ({min_class:,} per class)')

    else:
        print('❌ No EA data (2019-2020) found!')
    
    # Check Val dataset (2021 stress-test)
    raw_val = df_labeled[df_labeled['date'].dt.year == 2021]
    print(f'\n📊 RAW VAL DATA ANALYSIS (2021 STRESS-TEST INDIVIDUAL TWEETS):')
    print(f'Total raw Val tweets: {len(raw_val):,}')
    
    if len(raw_val) > 0:
        val_label_counts = raw_val['Label'].value_counts()
        print(f'Raw Val (2021) label counts:')
        for label, count in val_label_counts.items():
            pct = count / len(raw_val) * 100
            print(f'  {label}: {count:,} ({pct:.1f}%)')
        
        # Actionable signals for stress testing
        val_actionable = (raw_val['Label'] != 'Neutral').sum()
        val_actionable_pct = (val_actionable / len(raw_val) * 100)
        print(f'\nVal Actionable Signals:')
        print(f'  Actionable (non-neutral): {val_actionable:,} ({val_actionable_pct:.1f}%)')
        print(f'  Neutral: {len(raw_val) - val_actionable:,} ({100-val_actionable_pct:.1f}%)')
        print('✅ Full 2021 year available for out-of-sample stress testing')
    else:
        print('❌ No Val data (2021) found!')
    
    # Check EB dataset (2015-2018 & 2022-2023, excluding EA and Val years)
    raw_eb = df_labeled[
        ((df_labeled['date'].dt.year >= 2015) & (df_labeled['date'].dt.year <= 2018)) |
        ((df_labeled['date'].dt.year >= 2022) & (df_labeled['date'].dt.year <= 2023))
    ]
    print(f'\n📊 RAW EB DATA ANALYSIS (2015-2018 & 2022-2023 INDIVIDUAL TWEETS):')
    print(f'Total raw EB tweets: {len(raw_eb):,}')

    if len(raw_eb) > 0:
        # Label distribution
        eb_label_counts = raw_eb['Label'].value_counts()
        print(f'Raw EB label counts:')
        for label, count in eb_label_counts.items():
            pct = count / len(raw_eb) * 100
            print(f'  {label}: {count:,} ({pct:.1f}%)')
        
        # Event tweet analysis
        if 'Is_Event' in raw_eb.columns:
            event_tweets = raw_eb[raw_eb['Is_Event'] == 1]
            non_event_tweets = raw_eb[raw_eb['Is_Event'] == 0]
            print(f'\nEB Event Analysis:')
            print(f'  Event tweets: {len(event_tweets):,} ({len(event_tweets)/len(raw_eb)*100:.1f}%)')
            print(f'  Non-event tweets: {len(non_event_tweets):,} ({len(non_event_tweets)/len(raw_eb)*100:.1f}%)')
            
            # Check if we have enough event tweets for target
            event_target = 25000
            non_event_needed = 40000 - min(event_target, len(event_tweets))
            print(f'\nEB Sampling Strategy:')
            print(f'  Target event tweets: {min(event_target, len(event_tweets)):,} (max: {event_target:,})')
            print(f'  Target non-event tweets: {min(non_event_needed, len(non_event_tweets)):,}')
            total_eb_target = min(event_target, len(event_tweets)) + min(non_event_needed, len(non_event_tweets))
            print(f'  Total EB target: {total_eb_target:,} tweets')
            
            if len(event_tweets) >= event_target:
                print('✅ Sufficient event tweets for target sampling')
            else:
                shortage = event_target - len(event_tweets)
                print(f'⚠️  Event tweet shortage: need {shortage:,} more event tweets')
                
            if len(non_event_tweets) >= non_event_needed:
                print('✅ Sufficient non-event tweets to fill EB dataset')
            else:
                print(f'⚠️  Limited non-event tweets: only {len(non_event_tweets):,} available')
        else:
            print('⚠️  No Is_Event column found - all tweets treated as non-event')
        
        # Year breakdown for EB
        eb_year_counts = raw_eb['date'].dt.year.value_counts().sort_index()
        print(f'\nEB Year breakdown:')
        for year, count in eb_year_counts.items():
            pct = count / len(raw_eb) * 100
            print(f'  {year}: {count:,} tweets ({pct:.1f}%)')
            
        # Check temporal separation from EA and Val
        eb_years = set(raw_eb['date'].dt.year.unique())
        ea_val_years = {2019, 2020, 2021}
        overlap = eb_years & ea_val_years
        if overlap:
            print(f'❌ Year overlap detected between EB and EA/Val: {overlap}')
        else:
            print('✅ No year overlap between EB and EA/Val datasets')
            
        # Actionable signals analysis
        eb_actionable = (raw_eb['Label'] != 'Neutral').sum()
        eb_actionable_pct = (eb_actionable / len(raw_eb) * 100)
        print(f'\nEB Actionable Signals:')
        print(f'  Actionable (non-neutral): {eb_actionable:,} ({eb_actionable_pct:.1f}%)')
        print(f'  Neutral: {len(raw_eb) - eb_actionable:,} ({100-eb_actionable_pct:.1f}%)')
        
        if eb_actionable_pct >= 50:
            print('✅ Good actionable signal ratio for trading evaluation')
        else:
            print('⚠️  Low actionable signal ratio - may need adjustment')
            
    else:
        print('❌ No EB data found!')
        
    print(f'\n🎯 DATASET RESTRUCTURE & VOLATILITY MULTIPLIER IMPACT:')
    print(f'   EA Dataset: 2019-2020 training data (target: 90K tweets, ~30K per class)')
    print(f'   Val Dataset: 2021 full out-of-sample stress test (no size limit)')
    print(f'   EB Dataset: 2015-2018 & 2022-2023 event evaluation (excludes EA/Val years)')
    print(f'   Old config: fu_grid=[0.03, 0.05, 0.07] (too tight)')
    print(f'   New config: fu_grid=[0.5, 1.0, 1.5, 2.0] (proper volatility scaling)')
    print(f'   Expected: Better balanced labels, proper out-of-sample validation, meaningful signals')
    
    print(f'\n✅ Test completed! Results show impact of corrected volatility multipliers.')

if __name__ == "__main__":
    test_improved_labeling() 