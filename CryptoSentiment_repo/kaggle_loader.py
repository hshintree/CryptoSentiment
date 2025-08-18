from pathlib import Path
import pandas as pd
from dataset_loader import DatasetLoader

class KaggleLoader(DatasetLoader):
    """
    Specialized loader for Kaggle dataset (2021-2023) with rich metadata.
    Includes user metadata (followers, description) and tweet metadata,
    merged with price and volume data.
    """
    
    def load_dataset(self, *, aggregate: bool = False) -> pd.DataFrame:
        # Load the full dataset first
        df = super().load_dataset(aggregate=aggregate)
        
        # Filter to only Kaggle data
        df = df[df['data_source'] == 'kaggle'].copy()
        
        # Define columns to keep
        metadata_columns = [
            'date',              # timestamp
            'Tweet Content',     # the actual tweet
            'Close',            # price data
            'Volume',
            'log_volume',
            'RSI',             # technical indicators
            'ROC',
            'Is_Event',        # event flag
            # User metadata
            'user_description',
            'user_followers',
            'user_verified',
            # Tweet metadata
            'hashtags',
            'is_retweet',
        ]
        
        # Only keep columns that exist (some might be optional)
        existing_columns = [col for col in metadata_columns if col in df.columns]
        df = df[existing_columns]
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date')
        
        # Drop any rows with missing critical data
        required_columns = ['date', 'Tweet Content', 'Close', 'Volume', 'user_followers']
        df = df.dropna(subset=required_columns)
        
        return df
        
    def _load_prebit_dir(self) -> pd.DataFrame:
        """Override to return empty DataFrame since we don't want PreBit data"""
        return pd.DataFrame() 