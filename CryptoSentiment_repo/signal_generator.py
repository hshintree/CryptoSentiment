## signal_generator.py

import pandas as pd
import numpy as np
import yaml
from typing import Dict
from model import Model  # Assuming Model class from model.py

class SignalGenerator:
    """SignalGenerator class for converting model predictions into actionable trading signals."""

    def __init__(self, model: Model, config_path: str = 'config.yaml'):
        """
        Initialize the SignalGenerator with a trained model and configuration settings.

        Args:
            model (Model): The trained language model.
            config_path (str): Path to the configuration YAML file.
        """
        # Load configuration for signal generation settings
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = model
        self.methods = self.config['signal_generation'].get('methods', ['majority', 'mean'])
        self.rsi_threshold = self.config['data'].get('rsi_threshold', [30, 70])
        self.roc_window_length = self.config['data'].get('roc_window_length', 8)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions and aggregation methods.

        Args:
            data (pd.DataFrame): Preprocessed tweet data with technical indicators.

        Returns:
            pd.DataFrame: DataFrame with columns ['Date', 'Signal', 'Confidence'].
        """
        data['Predictions'] = self._predict(data)
        daily_signals = []

        for method in self.methods:
            if method == 'majority':
                signals_majority = self._generate_majority_signal(data)
                daily_signals.append(signals_majority)
            elif method == 'mean':
                signals_mean = self._generate_mean_signal(data)
                daily_signals.append(signals_mean)

        # Combine and return the best signal method(s) based on criteria
        combined_signals = self._combine_signals(daily_signals)
        return combined_signals

    def _predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Use the model to predict labels for each tweet in the dataset.

        Args:
            data (pd.DataFrame): Data containing tweets and related market data.

        Returns:
            np.ndarray: Array of predicted labels (0: Bearish, 1: Neutral, 2: Bullish).
        """
        predictions = []
        for _, row in data.iterrows():
            inputs = self.model.preprocess_input(
                tweet_content=row['Tweet Content'],
                rsi=row['RSI'],
                roc=row['ROC'],
                date=row['Tweet Date'],
                previous_label=row['Previous Label']
            )
            logits = self.model.forward(inputs)
            predicted_label = logits.argmax(dim=-1).item()
            predictions.append(predicted_label)
        return np.array(predictions)

    def _generate_majority_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using the majority vote method.

        Args:
            data (pd.DataFrame): Data with predictions.

        Returns:
            pd.DataFrame: DataFrame containing 'Date', 'Signal', and 'Confidence' columns.
        """
        results = []
        for date, group in data.groupby(data['Tweet Date']):
            signals_count = group['Predictions'].value_counts()
            dominant_signal = signals_count.idxmax()
            confidence = signals_count[dominant_signal] / signals_count.sum()
            results.append({'Date': date, 'Signal': dominant_signal, 'Confidence': confidence})
        
        return pd.DataFrame(results)

    def _generate_mean_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using the mean method.

        Args:
            data (pd.DataFrame): Data with predictions.

        Returns:
            pd.DataFrame: DataFrame containing 'Date', 'Signal', and 'Confidence' columns.
        """
        thresholds = self._optimization_for_thresholds()
        results = []
        for date, group in data.groupby(data['Tweet Date']):
            mean_value = group['Predictions'].mean()
            if mean_value < thresholds['bearish']:
                signal = 0  # Bearish
            elif mean_value > thresholds['bullish']:
                signal = 2  # Bullish
            else:
                signal = 1  # Neutral
            confidence = np.abs(mean_value - 1) / max(mean_value - thresholds['bearish'], thresholds['bullish'] - mean_value)
            results.append({'Date': date, 'Signal': signal, 'Confidence': confidence})
        
        return pd.DataFrame(results)

    def _optimization_for_thresholds(self) -> Dict[str, float]:
        """
        Optimization stubs for threshold determination.

        Returns:
            Dict[str, float]: Initialized threshold values.
        """
        # In a real scenario, dynamically optimize these values based on training and scoring cycles
        return {'bearish': 0.5, 'bullish': 1.5}

    def _combine_signals(self, signals_list: list) -> pd.DataFrame:
        """
        Combine signals from different methods into a cohesive output.

        Args:
            signals_list (list): List of DataFrames from different signal generation methods.

        Returns:
            pd.DataFrame: Combined DataFrame with the best signals and their confidences.
        """
        if len(signals_list) == 1:
            return signals_list[0]
        
        # Placeholder implementation to choose based on highest confidence
        combined_df = signals_list[0].set_index('Date')
        for df in signals_list[1:]:
            df = df.set_index('Date')
            combined_df = combined_df.join(df, lsuffix='_left', rsuffix='_right', how='outer')

        # Assume that we choose the signal with the higher confidence
        combined_df['Final_Signal'] = combined_df.apply(
            lambda row: row['Signal_left'] if row['Confidence_left'] > row['Confidence_right'] else row['Signal_right'], axis=1
        )
        combined_df['Final_Confidence'] = combined_df.apply(
            lambda row: max(row['Confidence_left'], row['Confidence_right']), axis=1
        )

        combined_df = combined_df.reset_index()
        combined_df = combined_df[['Date', 'Final_Signal', 'Final_Confidence']].rename(
            columns={'Final_Signal': 'Signal', 'Final_Confidence': 'Confidence'}
        )
        return combined_df

