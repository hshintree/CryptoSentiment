from pathlib import Path
import pandas as pd
from dataset_loader import DatasetLoader

class BigLoader(DatasetLoader):
    """
    Specialized loader for the complete 2015-2023 dataset.
    Focuses on core features only: tweets, prices, and volume.
    Combines both PreBit (2015-2021) and Kaggle (2021-2023) data.
    """
    
    def __init__(self, config_path_or_loader: str | "DatasetLoader" = "config.yaml", **kwargs):
        # Initialize parent class with config path
        print(f"BigLoader.__init__ called with: {config_path_or_loader}")
        super().__init__(config_path_or_loader, **kwargs)
        print("BigLoader initialization complete")
    
    def load_dataset(self, *, aggregate: bool = False) -> pd.DataFrame:
        # Load the full dataset first
        df = super().load_dataset(aggregate=aggregate)
        
        # Keep only the core columns we want
        core_columns = [
            'date',           # timestamp
            'Tweet Content',  # the actual tweet
            'Close',         # closing price
            'Volume',        # trading volume
            'log_volume',    # preprocessed log volume
            'RSI',          # technical indicators
            'ROC',
            'Is_Event',     # event flag
            'data_source'    # track data origin
        ]
        
        # Only keep columns that exist (some might be optional)
        existing_columns = [col for col in core_columns if col in df.columns]
        df = df[existing_columns]
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date')
        
        # Drop any rows with missing core data
        required_columns = ['date', 'Tweet Content', 'Close', 'Volume']
        df = df.dropna(subset=required_columns)
        
        return df 