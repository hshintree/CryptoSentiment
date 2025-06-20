import re
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nltk.stem import WordNetLemmatizer
import emoji

class Preprocessor:
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration file for preprocessing settings
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.settings = self.config['data']['preprocessing_steps']
        self.rsi_threshold   = self.config['data'].get('rsi_threshold', [30, 70])
        self.roc_window_length = self.config['data'].get('roc_window_length', 8)
        # lemmatizer once
        if self.settings.get('lemmatization', False):
            self.lemmatizer = WordNetLemmatizer()

        # prepare an (un-fitted) scaler for RSI/ROC
        self.scaler: Optional[MinMaxScaler] = None

    def _process_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process volume data to compute both z-score and log-volume.
        Uses a rolling window for z-score calculation to account for time-varying volume patterns.
        """
        if 'Volume' not in df.columns:
            return df
            
        # Compute log volume
        df['log_volume'] = np.log1p(df['Volume'])
        
        return df

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Text cleaning, RSI, ROC, volume, fill-na—but no scaling."""
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        ## 1) Text cleanup & normalization
        def clean_text(text: str) -> str:
            # 1a) unify case
            txt = text.lower()
            # 1b) strip URLs & @users
            txt = re.sub(r'http\S+|www\.\S+|@\w+', '', txt)
            # 1c) extract emojis (keep them as words)
            emojis = ' '.join(emoji.demojize(txt).split())
            # 1d) remove promotional markers (e.g. "buy now", "giveaway")
            promo_pattern = r'\b(buy now|giveaway|promo|free\s+gift)\b'
            emojis = re.sub(promo_pattern, '', emojis)
            # 1e) strip punctuation but keep hashtags
            emojis = re.sub(r'[^#\w\s]', '', emojis)
            # 1f) lemmatize
            if self.settings.get('lemmatization', False):
                tokens = []
                for w in emojis.split():
                    # preserve hashtags as tokens, but strip "#" before lemmatizing
                    if w.startswith('#'):
                        base = w[1:]
                        tokens.append('#' + self.lemmatizer.lemmatize(base))
                    else:
                        tokens.append(self.lemmatizer.lemmatize(w))
                emojis = ' '.join(tokens)
            return emojis

        data['Tweet Content'] = data['Tweet Content'].astype(str).apply(clean_text)

        ## 2) Technical indicators
        if 'Close' in data.columns:
            delta = data['Close'].diff(1)
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta).clip(lower=0).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['ROC'] = data['Close'].pct_change(periods=self.roc_window_length) * 100

        ## 3) Process volume data
        data = self._process_volume(data)

        ## 4) Fill missing values
        # Use forward fill first, then backward fill for any remaining NaNs
        data = data.ffill().bfill()
        
        return data

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy method for backward compatibility.
        WARNING: This applies global scaling and should not be used in fold-wise training!
        Use fit() and transform() instead for proper cross-validation.
        """
        data = self._compute_indicators(data)
        
        # ⚠️ GLOBAL SCALING - causes data leakage in cross-validation!
        if {'RSI','ROC'}.issubset(data.columns):
            scaler = MinMaxScaler()
            data[['RSI','ROC']] = scaler.fit_transform(data[['RSI','ROC']])
            
        return data

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the RSI/ROC scaler on df, return scaled df."""
        data = self._compute_indicators(df)
        if {'RSI','ROC'}.issubset(data.columns):
            self.scaler = MinMaxScaler().fit(data[['RSI','ROC']])
            data[['RSI','ROC']] = self.scaler.transform(data[['RSI','ROC']])
        return data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously‐fitted scaler to df."""
        if self.scaler is None:
            raise RuntimeError("Must call fit() before transform()")
        data = self._compute_indicators(df)
        data[['RSI','ROC']] = self.scaler.transform(data[['RSI','ROC']])
        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit() then transform()."""
        return self.fit(df)
