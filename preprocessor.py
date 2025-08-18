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
        # 12-day ROC default (paper)
        self.roc_window_length = self.config['data'].get('roc_window_length', 12)
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
            gain  = delta.clip(lower=0).rolling(window=14).mean()
            loss  = (-delta).clip(lower=0).rolling(window=14).mean()
            rs    = gain / loss

            # keep the **raw** 0‒100 RSI / %ROC that the paper discretises
            data['RSI_raw'] = 100 - (100 / (1 + rs))
            data['ROC_raw'] = (
                data['Close']
                    .pct_change(periods=self.roc_window_length)
                    .mul(100)
            )

        ## 3) Process volume data
        data = self._process_volume(data)

        ## 4) Fill missing values causally
        # Only forward-fill (propagate past into future), no backward-fill
        data = data.ffill()
        # Fill any initial NaNs in RSI/ROC with neutral defaults (no leakage)
        if 'RSI_raw' in data.columns:
            neutral_rsi = sum(self.rsi_threshold) / 2  # midpoint between thresholds, e.g. 50
            data['RSI_raw'] = data['RSI_raw'].fillna(neutral_rsi)
        if 'ROC_raw' in data.columns:
            data['ROC_raw'] = data['ROC_raw'].fillna(0.0)
        # Zero out any other residual NaNs (e.g. log_volume) safely
        data = data.fillna(0)
        
        ## 5) Dynamic ROC bucketing
        if 'ROC_raw' in data.columns:
            def _bucketise_roc(roc_series, window=8):
                rolling_std = roc_series.rolling(window).std()
                upper =  rolling_std
                lower = -rolling_std
                bucket = np.where(roc_series >  upper, "rising",
                         np.where(roc_series < lower, "falling", "neutral"))
                return bucket
            
            data['ROC_bucket'] = _bucketise_roc(data['ROC_raw'])
        
        return data

    # ⚠️ removed def preprocess to avoid accidental global scaling

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the RSI/ROC scaler on df, return scaled df."""
        data = self._compute_indicators(df)
        # scale *copies* so you can still access the raw values later
        if {'RSI_raw', 'ROC_raw'}.issubset(data.columns):
            self.scaler = MinMaxScaler().fit(data[['RSI_raw', 'ROC_raw']])
            data[['RSI', 'ROC']] = self.scaler.transform(
                data[['RSI_raw', 'ROC_raw']]
            )
        return data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously‐fitted scaler to df."""
        if self.scaler is None:
            raise RuntimeError("Must call fit() before transform()")
        data = self._compute_indicators(df)
        data[['RSI', 'ROC']] = self.scaler.transform(data[['RSI_raw', 'ROC_raw']])
        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit() then transform()."""
        return self.fit(df)
