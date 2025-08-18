#!/usr/bin/env python3
"""
Create labeled dataset for training by applying preprocessing and market labeling
to the aggregated dataset, with BERT token limit considerations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import re
from tqdm import tqdm
import yaml
from sklearn.model_selection import TimeSeriesSplit

from dataset_loader import DatasetLoader
from preprocessor import Preprocessor
from market_labeler import MarketLabeler


class LabeledDatasetCreator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.loader = DatasetLoader(config_path)
        self.preprocessor = Preprocessor(config_path)
        self.market_labeler = MarketLabeler(config_path)
        
        # BERT token limit considerations
        self.max_tokens_per_day = 512  # Conservative limit for BERT
        self.min_followers_threshold = 2000  # Filter out low-follower tweets
        
        # Check TBL config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        if "market_labeling" not in cfg or not cfg["market_labeling"]:
            raise ValueError("Triple-Barrier labeling parameters missing in config.yaml under 'market_labeling'.")
        
    def create_labeled_dataset(self, test_csv_path: str) -> pd.DataFrame:
        print(f"Loading aggregated dataset from: {test_csv_path}")
        df = pd.read_csv(test_csv_path, parse_dates=["date"])
        print(f"Loaded {len(df)} rows. Date range: {df['date'].min()} to {df['date'].max()}")

        # 1. Filter for BERT token limits (keep existing logic if present)
        df_filtered = self._filter_tweets_for_bert(df)
        print(f"After BERT filtering: {len(df_filtered)} rows")

        # 2. Preprocess
        print("Applying preprocessing...")
        df_preprocessed = self.preprocessor.preprocess(df_filtered)
        print("Preprocessing complete.")

        # 3. Market labeling (Triple-Barrier)
        print("Applying Triple-Barrier market labeling...")
        df_labeled = self.market_labeler.label_data(df_preprocessed)
        print("Market labeling complete.")

        # 4. Aggregate per day (if needed)
        if df_labeled['date'].dt.time.nunique() > 1:
            print("Aggregating multiple tweets per day...")
            df_labeled = self._aggregate_per_day(df_labeled)
            print(f"After aggregation: {len(df_labeled)} rows")

        # 5. Validate
        print("Validating labeled dataset...")
        if df_labeled['Label'].isna().any():
            raise ValueError("Some rows are missing market labels after Triple-Barrier labeling.")
        if df_labeled['Close'].isna().any():
            raise ValueError("Some rows are missing Close price after aggregation.")
        print("Validation passed: all rows have Label and Close.")

        # 6. Save
        out_path = Path("data/labeled_test.csv")
        df_labeled.to_csv(out_path, index=False)
        print(f"Saved labeled dataset to {out_path} ({len(df_labeled)} rows)")
        return df_labeled

    def _filter_tweets_for_bert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter tweets to handle BERT token limits by:
        1. Filtering out low-follower tweets from Kaggle dataset
        2. Truncating very long tweet content
        3. Ensuring we don't exceed token limits per day
        """
        print("Filtering tweets for BERT token limits...")
        
        # Make a copy to avoid modifying original
        df_filtered = df.copy()
        
        # Step 1: Filter by user followers for Kaggle tweets
        if 'user_followers' in df_filtered.columns:
            print(f"Filtering Kaggle tweets by follower count (threshold: {self.min_followers_threshold})")
            
            # Count tweets before filtering
            total_tweets = len(df_filtered)
            kaggle_tweets = df_filtered[df_filtered['data_source'] == 'kaggle']
            prebit_tweets = df_filtered[df_filtered['data_source'] == 'prebit']
            
            print(f"Before filtering: {len(kaggle_tweets)} Kaggle tweets, {len(prebit_tweets)} PreBit tweets")
            
            # Filter Kaggle tweets by follower count
            high_follower_mask = (
                (df_filtered['data_source'] == 'kaggle') & 
                (df_filtered['user_followers'] >= self.min_followers_threshold)
            ) | (df_filtered['data_source'] == 'prebit')  # Keep all PreBit tweets
            
            df_filtered = df_filtered[high_follower_mask]
            
            kaggle_after = len(df_filtered[df_filtered['data_source'] == 'kaggle'])
            prebit_after = len(df_filtered[df_filtered['data_source'] == 'prebit'])
            
            print(f"After filtering: {kaggle_after} Kaggle tweets, {prebit_after} PreBit tweets")
            print(f"Removed {total_tweets - len(df_filtered)} low-follower tweets")
        
        # Step 2: Truncate very long tweet content
        print("Truncating very long tweet content...")
        max_chars = 280  # Twitter character limit as rough proxy for tokens
        
        def truncate_tweet(text):
            if pd.isna(text) or len(str(text)) <= max_chars:
                return text
            return str(text)[:max_chars] + "..."
        
        df_filtered['Tweet Content'] = df_filtered['Tweet Content'].apply(truncate_tweet)
        
        # Step 3: Estimate token count and filter if needed
        print("Estimating token counts...")
        df_filtered['estimated_tokens'] = df_filtered['Tweet Content'].apply(
            lambda x: len(str(x).split()) * 1.3  # Rough estimate: words * 1.3 for BERT tokens
        )
        
        # Check for extremely long tweets that might exceed BERT limits
        very_long_mask = df_filtered['estimated_tokens'] > 400  # Conservative limit
        if very_long_mask.sum() > 0:
            print(f"Found {very_long_mask.sum()} tweets with >400 estimated tokens")
            print("Truncating these tweets further...")
            
            def truncate_for_bert(text):
                words = str(text).split()
                if len(words) > 300:  # Conservative word limit
                    return ' '.join(words[:300]) + "..."
                return text
            
            df_filtered.loc[very_long_mask, 'Tweet Content'] = df_filtered.loc[very_long_mask, 'Tweet Content'].apply(truncate_for_bert)
            df_filtered.loc[very_long_mask, 'estimated_tokens'] = df_filtered.loc[very_long_mask, 'Tweet Content'].apply(
                lambda x: len(str(x).split()) * 1.3
            )
        
        # Step 4: Aggregate tweets per day to ensure we don't exceed daily token limits
        print("Aggregating tweets per day to manage token limits...")
        df_daily = self._aggregate_daily_with_token_limit(df_filtered)
        
        print(f"Daily aggregation complete: {len(df_daily)} days")
        print(f"Average tokens per day: {df_daily['estimated_tokens'].mean():.1f}")
        print(f"Max tokens per day: {df_daily['estimated_tokens'].max():.1f}")
        
        return df_daily
    
    def _aggregate_daily_with_token_limit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate tweets per day while respecting BERT token limits.
        If a day has too many tweets, prioritize by:
        1. User followers (higher = better)
        2. Tweet length (shorter = more tokens available for others)
        """
        print("Aggregating tweets per day with token limit consideration...")
        
        daily_groups = []
        
        for date, day_df in tqdm(df.groupby('date'), desc="Processing days"):
            day_tweets = day_df.copy().sort_values('date')
            
            # If we have too many tweets for this day, prioritize
            total_tokens = day_tweets['estimated_tokens'].sum()
            
            if total_tokens > self.max_tokens_per_day:
                print(f"Day {date}: {len(day_tweets)} tweets, {total_tokens:.1f} tokens - prioritizing...")
                
                # Sort by priority: higher followers first, then shorter tweets
                day_tweets = day_tweets.sort_values(
                    ['user_followers', 'estimated_tokens'], 
                    ascending=[False, True]
                )
                
                # Select tweets until we hit the token limit
                selected_tweets = []
                current_tokens = 0
                
                for _, tweet in day_tweets.iterrows():
                    tweet_tokens = tweet['estimated_tokens']
                    if current_tokens + tweet_tokens <= self.max_tokens_per_day:
                        selected_tweets.append(tweet)
                        current_tokens += tweet_tokens
                    else:
                        break
                
                day_tweets = pd.DataFrame(selected_tweets)
                print(f"  Selected {len(day_tweets)} tweets, {current_tokens:.1f} tokens")
            
            # Aggregate the selected tweets for this day
            if len(day_tweets) > 0:
                aggregated_day = self._aggregate_single_day(day_tweets)
                daily_groups.append(aggregated_day)
        
        if daily_groups:
            result = pd.concat(daily_groups, ignore_index=True)
            return result
        else:
            return pd.DataFrame()
    
    def _aggregate_single_day(self, day_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple tweets for a single day into one record.
        """
        # Combine tweet content
        combined_content = " [SEP] ".join(day_df['Tweet Content'].astype(str))
        
        # Use the first non-null value for other columns
        first_row = day_df.iloc[0].copy()
        first_row['Tweet Content'] = combined_content
        
        # Sum up estimated tokens
        first_row['estimated_tokens'] = day_df['estimated_tokens'].sum()
        
        # Update metadata
        if 'data_source' in day_df.columns:
            sources = day_df['data_source'].unique()
            first_row['data_source'] = '+'.join(sources)
        
        if 'user_followers' in day_df.columns:
            # Use max followers as representative
            first_row['user_followers'] = day_df['user_followers'].max()
        
        return pd.DataFrame([first_row])
    
    def _aggregate_per_day(self, df: pd.DataFrame) -> pd.DataFrame:
        # Group by date (date only, not datetime)
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.date
        agg = df.groupby('date').agg({
            'Tweet Content': lambda x: ' [SEP] '.join(x.astype(str)),
            'Close': 'first',
            'Volume': 'first',
            'RSI': 'first' if 'RSI' in df.columns else 'first',
            'ROC': 'first' if 'ROC' in df.columns else 'first',
            'Label': 'first',
        }).reset_index()
        agg['date'] = pd.to_datetime(agg['date'])
        return agg


class EvalLoader:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.df = pd.read_csv("data/combined_dataset_raw.csv")
        self.df['date'] = pd.to_datetime(self.df['date'])
        if 'Is_Event' not in self.df.columns:
            raise ValueError("Is_Event column missing from dataset.")
        self.labeler = MarketLabeler(config_path)

    def create_eval_datasets(self, n_folds=5):
        # 1. Label the data
        df = self.labeler.label_data(self.df)
        
        # 2. Create Ea and Eb datasets
        ea_df = self._create_ea_dataset(df)
        eb_df = self._create_eb_dataset(df)
        
        # 3. Time-series splits
        ea_train_folds, ea_test_folds = self._create_time_folds(ea_df, n_folds)
        eb_train_folds, eb_test_folds = self._create_time_folds(eb_df, n_folds)
        
        # 4. Daily aggregate
        all_data = self._aggregate_all_data([ea_df, eb_df])
        
        return {
            "ea_train_folds": ea_train_folds,
            "ea_test_folds":  ea_test_folds,
            "eb_train_folds": eb_train_folds,
            "eb_test_folds":  eb_test_folds,
            "ea_full":        ea_df,
            "eb_full":        eb_df,
            "all_data":       all_data
        }

    def _create_ea_dataset(self, df):
        # Filter for 2020
        ea = df[df['date'].dt.year == 2020].copy()
        # Drop rows with missing labels
        ea = ea[ea['Label'].isin(['Bullish', 'Bearish', 'Neutral'])]
        # Sample 20k of each class
        sampled = []
        for label in ['Bullish', 'Bearish', 'Neutral']:
            group = ea[ea['Label'] == label]
            n = min(20000, len(group))
            sampled.append(group.sample(n=n, random_state=42))
        ea_sampled = pd.concat(sampled).sort_values('date')
        ea_sampled = self._add_previous_day_label(ea_sampled)
        return ea_sampled

    def _create_eb_dataset(self, df):
        # Filter for 2015–2019 & 2021–2023
        eb = df[((df['date'].dt.year >= 2015) & (df['date'].dt.year <= 2019)) |
                ((df['date'].dt.year >= 2021) & (df['date'].dt.year <= 2023))].copy()
        # Event tweets
        event_tweets = eb[eb['Is_Event'] == 1]
        n_event = min(25000, len(event_tweets))
        event_sample = event_tweets.sample(n=n_event, random_state=42)
        # Non-event tweets to fill up to ~40k
        non_event_tweets = eb[eb['Is_Event'] == 0]
        n_non_event = min(40000 - n_event, len(non_event_tweets))
        non_event_sample = non_event_tweets.sample(n=n_non_event, random_state=42)
        eb_sampled = pd.concat([event_sample, non_event_sample]).sort_values('date')
        eb_sampled = self._add_previous_day_label(eb_sampled)
        return eb_sampled

    def _create_time_folds(self, df, n_folds):
        tscv = TimeSeriesSplit(n_splits=n_folds)
        train_folds, test_folds = [], []
        df = df.sort_values('date').reset_index(drop=True)
        for train_idx, test_idx in tscv.split(df):
            train_folds.append(df.iloc[train_idx].copy())
            test_folds.append(df.iloc[test_idx].copy())
        return train_folds, test_folds

    def _add_previous_day_label(self, df):
        df = df.sort_values('date').reset_index(drop=True)
        df['Previous_Label'] = df['Label'].shift(1).fillna('Neutral')
        return df

    def _aggregate_all_data(self, dfs):
        # Concatenate and group by date
        all_df = pd.concat(dfs).sort_values('date')
        # Group by date, aggregate
        def agg_func(x):
            return x.mode()[0] if not x.mode().empty else x.iloc[0]
        grouped = all_df.groupby(all_df['date'].dt.date).agg({
            'Label': agg_func,
            'Close': 'mean',
            'Volume': 'mean' if 'Volume' in all_df.columns else 'first',
            'Tweet Content': lambda x: ' [SEP] '.join(x.astype(str)),
            'Is_Event': 'max'
        }).reset_index().rename(columns={'index': 'date'})
        grouped['date'] = pd.to_datetime(grouped['date'])
        grouped = grouped.sort_values('date').reset_index(drop=True)
        grouped = self._add_previous_day_label(grouped)
        return grouped
    

def main():
    """Main function to create and test EvalLoader datasets."""
    from eval_loader import EvalLoader
    
    csv_path = "data/combined_dataset_raw.csv"
    loader = EvalLoader(csv_path, config_path="config.yaml")
    result = loader.create_eval_datasets(n_folds=5)
    
    print("\n" + "="*50)
    print("EVALUATION DATASETS SUMMARY")
    print("="*50)
    
    print(f"Ea train folds: {[len(f) for f in result['ea_train_folds']]}")
    print(f"Ea test folds: {[len(f) for f in result['ea_test_folds']]}")
    print(f"Eb train folds: {[len(f) for f in result['eb_train_folds']]}")
    print(f"Eb test folds: {[len(f) for f in result['eb_test_folds']]}")
    print(f"Ea full shape: {result['ea_full'].shape}")
    print(f"Eb full shape: {result['eb_full'].shape}")
    print(f"All data shape: {result['all_data'].shape}")
    
    print("\nEa dataset sample:")
    print(result['ea_full'][['date', 'Label', 'Previous_Label']].head())
    
    print("\nEb dataset sample:")
    print(result['eb_full'][['date', 'Label', 'Is_Event', 'Previous_Label']].head())
    
    print("\nAggregated data sample:")
    print(result['all_data'][['date', 'Label', 'Is_Event', 'Previous_Label']].head())


if __name__ == "__main__":
    main() 