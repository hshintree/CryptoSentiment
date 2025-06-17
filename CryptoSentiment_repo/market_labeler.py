## market_labeler.py

import pandas as pd
import numpy as np
import yaml
from typing import Dict

class MarketLabeler:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the labeler with configuration settings."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        market_labeling_config = self.config.get('market_labeling', {})
        self.strategy = market_labeling_config.get('strategy', 'TBL')
        self.vertical_barrier_range = market_labeling_config.get('barrier_window', '8-15')
        self.vertical_barrier_min, self.vertical_barrier_max = map(int, self.vertical_barrier_range.split('-'))
        # Multipliers applied to volatility when computing the price barriers
        self.upper_vol_mult = market_labeling_config.get('upper_vol_mult', 1.0)
        self.lower_vol_mult = market_labeling_config.get('lower_vol_mult', 1.0)

    def label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Triple Barrier Labeling with volatility scaling."""
        volatility = self.estimate_historical_volatility(data).fillna(method='bfill').fillna(method='ffill')

        labels = []
        upper_list = []
        lower_list = []
        vert_list = []

        for i in range(len(data)):
            window = np.random.randint(self.vertical_barrier_min, self.vertical_barrier_max + 1)
            end_idx = min(i + window, len(data) - 1)

            start_price = data.at[i, 'Close']
            up = start_price * (1 + volatility.iloc[i] * self.upper_vol_mult)
            down = start_price * (1 - volatility.iloc[i] * self.lower_vol_mult)

            price_path = data.loc[i:end_idx, 'Close']

            bullish = (price_path >= up).any()
            bearish = (price_path <= down).any()

            if bullish and not bearish:
                label = 'Bullish'
            elif bearish and not bullish:
                label = 'Bearish'
            else:
                label = 'Neutral'

            labels.append(label)
            upper_list.append(up)
            lower_list.append(down)
            vert_list.append(end_idx)

        data = data.copy()
        data['Upper Barrier'] = upper_list
        data['Lower Barrier'] = lower_list
        data['Vertical Barrier'] = vert_list
        data['Label'] = labels

        return data

    def estimate_historical_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Estimate historical volatility using EWMA on log returns."""
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        volatility = log_returns.ewm(span=30, adjust=False).std()
        return volatility

