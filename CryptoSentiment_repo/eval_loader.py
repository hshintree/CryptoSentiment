from pathlib import Path
from typing import Union, TYPE_CHECKING
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import TimeSeriesSplit
from big_loader import BigLoader
from market_labeler import MarketLabeler

if TYPE_CHECKING:
    from big_loader import BigLoader

class EvalLoader(BigLoader):
    """
    Specialized loader for creating evaluation datasets Ea and Eb.
    
    Ea: In-sample 2020 dataset with ~60k balanced tweets
    Eb: Out-of-sample 2015-2023 event-focused dataset with ~40k tweets
    """
    
    def __init__(self, config_path_or_loader: Union[str, BigLoader] = "config.yaml", **kwargs):
        print(f"EvalLoader.__init__ called with: {config_path_or_loader}")
        super().__init__(config_path_or_loader, **kwargs)
        self.labeler = MarketLabeler()
        print("EvalLoader initialization complete")
        
    def create_eval_datasets(self, n_folds: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Create both Ea and Eb evaluation datasets with their respective folds.
        Returns a dictionary containing:
        - ea_train_folds: list of DataFrames for each Ea training fold
        - ea_test_folds: list of DataFrames for each Ea testing fold
        - eb_train_folds: list of DataFrames for each Eb training fold
        - eb_test_folds: list of DataFrames for each Eb testing fold
        - ea_full: complete Ea dataset
        - eb_full: complete Eb dataset
        - all_data: merged daily-aggregated dataset if aggregate=True
        """
        # Load and preprocess all data
        df = super().load_dataset(aggregate=False)
        
        # Add market-derived labels using triple-barrier method
        df = self.labeler.label_data(df)
        
        # Create Ea (2020 in-sample)
        ea_df = self._create_ea_dataset(df)
        ea_train_folds, ea_test_folds = self._create_time_folds(ea_df, n_folds)
        
        # Create Eb (2015-2023 event-focused, excluding 2020)
        eb_df = self._create_eb_dataset(df)
        eb_train_folds, eb_test_folds = self._create_time_folds(eb_df, n_folds)
        
        # Create aggregated version if requested
        all_data = self._aggregate_all_data(pd.concat([ea_df, eb_df]))
        
        return {
            'ea_train_folds': ea_train_folds,
            'ea_test_folds': ea_test_folds,
            'eb_train_folds': eb_train_folds,
            'eb_test_folds': eb_test_folds,
            'ea_full': ea_df,
            'eb_full': eb_df,
            'all_data': all_data
        }
    
    def _create_ea_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the Ea (2020 in-sample) dataset with balanced labels."""
        # Filter for 2020
        mask = df['date'].dt.year == 2020
        ea_df = df[mask].copy()
        
        # Sample ~60k tweets evenly across labels
        target_per_class = 20000  # 60k total
        balanced_samples = []
        for label in ['Bullish', 'Bearish', 'Neutral']:
            class_df = ea_df[ea_df['Label'] == label]
            if len(class_df) > target_per_class:
                class_df = class_df.sample(n=target_per_class, random_state=42)
            balanced_samples.append(class_df)
        
        ea_df = pd.concat(balanced_samples)
        ea_df = self._add_previous_day_label(ea_df)
        return ea_df.sort_values('date')
    
    def _create_eb_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the Eb (event-focused) dataset excluding 2020."""
        # Exclude 2020
        mask = df['date'].dt.year != 2020
        eb_df = df[mask].copy()
        
        # Split into event and non-event days
        event_tweets = eb_df[eb_df['Is_Event'] == 1]
        non_event_tweets = eb_df[eb_df['Is_Event'] == 0]
        
        # Sample from event days (prioritize these)
        event_sample = event_tweets.sample(n=min(25000, len(event_tweets)), random_state=42)
        
        # Sample from non-event days to balance
        remaining_samples = 40000 - len(event_sample)
        non_event_sample = non_event_tweets.sample(n=remaining_samples, random_state=42)
        
        # Combine and add previous day label
        eb_df = pd.concat([event_sample, non_event_sample])
        eb_df = self._add_previous_day_label(eb_df)
        return eb_df.sort_values('date')
    
    def _create_time_folds(self, df: pd.DataFrame, n_folds: int) -> Tuple[list, list]:
        """Create time-series cross-validation folds preserving date order."""
        tscv = TimeSeriesSplit(n_splits=n_folds)
        train_folds, test_folds = [], []
        
        for train_idx, test_idx in tscv.split(df):
            train_folds.append(df.iloc[train_idx])
            test_folds.append(df.iloc[test_idx])
            
        return train_folds, test_folds
    
    def _add_previous_day_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add previous day's label for each tweet."""
        df = df.sort_values('date')
        df['Previous_Label'] = df.groupby(df['date'].dt.date)['Label'].transform(
            lambda x: x.shift(1).fillna(method='ffill')
        )
        # Fill first day's labels with 'Neutral'
        df['Previous_Label'] = df['Previous_Label'].fillna('Neutral')
        return df
    
    def _aggregate_all_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tweets by day with majority label and mean numeric features."""
        # Numeric columns to average
        numeric_cols = ['Close', 'RSI', 'ROC', 'log_volume']
        
        # Group by date
        grouped = df.groupby(df['date'].dt.date).agg({
            'Label': lambda x: x.mode().iloc[0],  # majority label
            'Tweet Content': lambda x: ' \n'.join(x),  # concatenate tweets
            'Is_Event': 'max',  # if any tweet that day was an event
            **{col: 'mean' for col in numeric_cols}  # average numeric features
        }).reset_index()
        
        # Add previous day label
        grouped = self._add_previous_day_label(grouped)
        
        return grouped 