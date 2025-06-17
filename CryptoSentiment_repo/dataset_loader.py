## dataset_loader.py

import os
import pandas as pd
import yaml

class DatasetLoader:
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Extract paths from configuration
        self.bitcoin_events_path = self.config['data'].get('bitcoin_events_path', 'path/to/bitcoin_historical_events.csv')
        self.tweets_data_path = self.config['data'].get('tweets_data_path', 'path/to/tweet_data.csv')
        self.output_data_path = self.config['data'].get('output_data_path', 'path/to/output')

        # Validate paths
        if not os.path.exists(self.bitcoin_events_path):
            raise FileNotFoundError(f"Bitcoin events file not found at path: {self.bitcoin_events_path}")
            
        if not os.path.exists(self.tweets_data_path):
            raise FileNotFoundError(f"Tweets data file not found at path: {self.tweets_data_path}")

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
