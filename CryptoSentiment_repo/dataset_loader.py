## dataset_loader.py

import os
import pandas as pd
import yaml

class DatasetLoader:
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Extract paths from configuration. The repository can either use the
        # original split between events and tweets or the combined PreBit
        # dataset.  ``prebit_dataset_path`` takes precedence if provided. If the
        # dataset is split across multiple CSVs provide ``prebit_dataset_dir``.
        self.prebit_dataset_path = self.config['data'].get('prebit_dataset_path')
        self.prebit_dataset_dir = self.config['data'].get('prebit_dataset_dir')
        self.bitcoin_events_path = self.config['data'].get(
            'bitcoin_events_path', 'path/to/bitcoin_historical_events.csv')
        self.tweets_data_path = self.config['data'].get(
            'tweets_data_path', 'path/to/tweet_data.csv')
        self.output_data_path = self.config['data'].get(
            'output_data_path', 'path/to/output')

        # Validate paths. If a PreBit dataset path or directory is provided,
        # skip the individual event/tweet checks.
        if self.prebit_dataset_path:
            if not os.path.exists(self.prebit_dataset_path):
                raise FileNotFoundError(
                    f"PreBit dataset file not found at path: {self.prebit_dataset_path}")
        elif self.prebit_dataset_dir:
            if not os.path.isdir(self.prebit_dataset_dir):
                raise FileNotFoundError(
                    f"PreBit dataset directory not found at path: {self.prebit_dataset_dir}")
        else:
            if not os.path.exists(self.bitcoin_events_path):
                raise FileNotFoundError(
                    f"Bitcoin events file not found at path: {self.bitcoin_events_path}")

            if not os.path.exists(self.tweets_data_path):
                raise FileNotFoundError(
                    f"Tweets data file not found at path: {self.tweets_data_path}")

    def load_event_data(self) -> pd.DataFrame:
        """Load Bitcoin historical events data"""
        try:
            # Load data using pandas
            event_data = pd.read_csv(self.bitcoin_events_path, parse_dates=['Event Date'])
            # Ensure necessary columns are present
            if 'Event Date' not in event_data.columns or 'Event Description' not in event_data.columns:
                raise ValueError("Required columns missing from the Bitcoin events data.")
            
            # Return the loaded dataframe
            return event_data
        except Exception as e:
            raise RuntimeError(f"Failed to load Bitcoin historical events data: {e}")

    def load_tweet_data(self) -> pd.DataFrame:
        """Load Tweets data"""
        try:
            # Load data using pandas
            tweet_data = pd.read_csv(self.tweets_data_path, parse_dates=['Tweet Date'])
            # Basic check for minimum expected columns
            if 'Tweet Date' not in tweet_data.columns or 'Tweet Content' not in tweet_data.columns:
                raise ValueError("Required columns missing from the Tweets data.")
            
            # Return the loaded dataframe
            return tweet_data
        except Exception as e:
            raise RuntimeError(f"Failed to load Tweets data: {e}")

    def load_prebit_data(self) -> pd.DataFrame:
        """Load the PreBit multimodal dataset."""
        if self.prebit_dataset_path:
            return self._load_single_prebit()
        if self.prebit_dataset_dir:
            return self._load_split_prebit()
        raise RuntimeError("PreBit dataset information not provided in config")

    def _load_single_prebit(self) -> pd.DataFrame:
        """Load a single CSV version of the PreBit dataset."""
        try:
            data = pd.read_csv(self.prebit_dataset_path, parse_dates=['date'])
            required = {'date', 'tweet', 'close'}
            if not required.issubset(set(data.columns)):
                raise ValueError(
                    "PreBit dataset must contain columns: 'date', 'tweet', 'close'")
            data = data.rename(columns={
                'date': 'Tweet Date',
                'tweet': 'Tweet Content',
                'close': 'Close'
            })
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load PreBit dataset: {e}")

    def _load_split_prebit(self) -> pd.DataFrame:
        """Load the PreBit dataset when tweets and prices are stored separately."""
        import glob

        tweet_files = sorted(glob.glob(os.path.join(self.prebit_dataset_dir, 'combined_tweets_*_labeled.csv')))
        if not tweet_files:
            raise FileNotFoundError('No tweet CSV files found in the PreBit dataset directory')

        tweet_frames = []
        for path in tweet_files:
            df = pd.read_csv(path, parse_dates=['date'])
            if 'text_split' not in df.columns:
                raise ValueError('Tweet CSV missing required columns')
            tweet_frames.append(df[['date', 'text_split']].rename(columns={'date': 'Tweet Date', 'text_split': 'Tweet Content'}))
        tweets = pd.concat(tweet_frames, ignore_index=True)

        price_path = os.path.join(self.prebit_dataset_dir, 'price_label.csv')
        if os.path.exists(price_path):
            price_df = pd.read_csv(price_path, parse_dates=['date'])
            close_col = 'Close_x' if 'Close_x' in price_df.columns else 'close'
            price_df = price_df.rename(columns={'date': 'Tweet Date', close_col: 'Close'})
            price_df = price_df[['Tweet Date', 'Close']]
            data = tweets.merge(price_df, on='Tweet Date', how='left')
        else:
            data = tweets

        return data
