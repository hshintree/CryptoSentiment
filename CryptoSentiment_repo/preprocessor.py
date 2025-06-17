## preprocessor.py

import pandas as pd
import numpy as np
import yaml
from typing import Dict
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration file for preprocessing settings
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Extract preprocessing settings
        self.settings = self.config['data']['preprocessing_steps']
        self.rsi_threshold = self.config['data'].get('rsi_threshold', [30, 70])
        self.roc_window_length = self.config['data'].get('roc_window_length', 8)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data according to the specified steps"""

        # Text Normalization
        if self.settings['text_normalization']:
            data['Tweet Content'] = data['Tweet Content'].str.lower()

        if self.settings['remove_urls']:
            data['Tweet Content'] = data['Tweet Content'].str.replace(r'http\S+|www.\S+', '', regex=True)

        if self.settings['remove_user_ids']:
            data['Tweet Content'] = data['Tweet Content'].str.replace(r'@\w+', '', regex=True)
        
        if self.settings['remove_punctuation']:
            data['Tweet Content'] = data['Tweet Content'].str.replace(r'[^\w\s]', '', regex=True)

        if self.settings['lemmatization']:
            # Assuming lemmatization method is defined somewhere else
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            data['Tweet Content'] = data['Tweet Content'].apply(
                lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
            )

        # Ensure 'Close' price exists in case technical indicators are calculated
        if 'Close' in data.columns:
            # RSI calculation
            delta = data['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Rate of Change (ROC) calculation
            data['ROC'] = data['Close'].pct_change(periods=self.roc_window_length) * 100

        # Handle missing values
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Min-Max normalizing tech indicators for embedding in prompts if necessary
        scaler = MinMaxScaler()
        if 'RSI' in data.columns and 'ROC' in data.columns:
            data[['RSI', 'ROC']] = scaler.fit_transform(data[['RSI', 'ROC']])

        return data
