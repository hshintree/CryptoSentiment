## market_labeler.py

import pandas as pd
import numpy as np
import yaml
from typing import Dict

class MarketLabeler:
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Extract settings from configuration
        market_labeling_config = self.config.get('market_labeling', {})
        self.strategy = market_labeling_config.get('strategy', 'TBL')
        self.vertical_barrier_range = market_labeling_config.get('barrier_window', '8-15')
        self.vertical_barrier_min, self.vertical_barrier_max = map(int, self.vertical_barrier_range.split('-'))

    def label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the Triple Barrier Labeling to assign market-driven labels."""
        volatility = self.estimate_historical_volatility(data)
        data['Upper Barrier'] = data['Close'] + (volatility * self.upper_barrier_factor(data))
        data['Lower Barrier'] = data['Close'] - (volatility * self.lower_barrier_factor(data))
        data['Vertical Barrier'] = self.vertical_barrier_max

        # Assign labels based on barrier conditions
        data['Label'] = self.assign_labels(data)

        return data

    def estimate_historical_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Estimate historical volatility using EWMA on log returns."""
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        volatility = log_returns.ewm(span=30, adjust=False).std()
        return volatility

    def upper_barrier_factor(self, data: pd.DataFrame) -> float:
        """Calculate the factor for the upper barrier based on market conditions."""
        # Placeholder for a potentially complex decision algorithm
        return 1.0

    def lower_barrier_factor(self, data: pd.DataFrame) -> float:
        """Calculate the factor for the lower barrier based on market conditions."""
        # Placeholder for a potentially complex decision algorithm
        return 1.0

    def assign_labels(self, data: pd.DataFrame) -> pd.Series:
        """Assign labels based on the first barrier touched within the vertical barrier limit."""
        labels = []
        for i in range(len(data)):
            period_data = data.iloc[i:min(i + self.vertical_barrier_max, len(data))]
            price_path = period_data['Close']

            upper_barrier = data.at[i, 'Upper Barrier']
            lower_barrier = data.at[i, 'Lower Barrier']

            bullish = (price_path >= upper_barrier).any()
            bearish = (price_path <= lower_barrier).any()

            if bullish and not bearish:
                labels.append('Bullish')
            elif bearish and not bullish:
                labels.append('Bearish')
            else:
                labels.append('Neutral')
            
        return pd.Series(labels, index=data.index)
