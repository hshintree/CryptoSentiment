## backtester.py

import vectorbt as vbt
import pandas as pd
import yaml
from typing import Dict, Any


class Backtester:
    """Backtester uses vectorbt to simulate trading strategies and evaluate their performances."""

    def __init__(self, signals: pd.DataFrame, config_path: str = 'config.yaml'):
        """
        Initialize Backtester with trading signals and configuration settings.
        
        Args:
            signals (pd.DataFrame): DataFrame containing trading signals with columns ['Date', 'Signal', 'Confidence'].
            config_path (str): Path to the configuration yaml file.
        """
        # Load configuration file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.signals = signals.set_index('Date')  # Ensure Date is the index for vectorbt compatibility

        # Extract market regime periods from config if specified
        self.market_regimes = {
            'Bullish': pd.date_range('2017-01-01', '2018-12-31'),
            'Bearish': pd.date_range('2018-01-01', '2019-12-31'),
            'Neutral': pd.date_range('2019-01-01', '2020-06-30')
        }

    def run_backtest(self) -> Dict[str, Any]:
        """
        Execute backtesting across defined regimes and evaluate performance metrics.
        
        Returns:
            dict: A dictionary containing the performance metrics for each market regime and overall strategy.
        """
        # Market Price Data Placeholder
        price_data = self._get_market_data()

        # Running simulations (backtest)
        results = {}
        for regime, date_range in self.market_regimes.items():
            print(f"Running backtest for {regime} market regime.")
            
            # Filtering data specific to the date range per market regime
            signals = self.signals.loc[date_range]
            prices = price_data.loc[date_range]

            # Creating strategy analysis
            portfolio = vbt.Portfolio.from_signals(
                prices['Close'],
                signals.loc[:, 'Signal'] == 2,  # Long signal
                signals.loc[:, 'Signal'] == 0,  # Short signal
                short_conf_price=prices['Close'],
                close_first=False
            )

            metrics = self.get_performance_metrics(portfolio, regime)
            results[regime] = metrics

        return results

    def _get_market_data(self) -> pd.DataFrame:
        """
        Placeholder function to demonstrate market data fetching.
        
        Returns:
            pd.DataFrame: A dummy DataFrame of market Close prices for the backtesting periods.
        """
        # Example: Create dummy data for Close prices indexed by a full date range
        full_date_range = pd.date_range('2017-01-01', '2020-06-30')
        close_prices = pd.Series(100 + 0.01 * (full_date_range - full_date_range[0]).days, index=full_date_range)
        return pd.DataFrame({'Close': close_prices})

    def get_performance_metrics(self, portfolio, regime: str) -> Dict[str, float]:
        """
        Calculate performance metrics of a backtested strategy.
        
        Args:
            portfolio: Vectorbt portfolio object summarizing strategy performance
            regime (str): Current market regime name for context
        
        Returns:
            dict: Dictionary of calculated metrics from the portfolio.
        """
        # Extracting usual trading metrics from the portfolio
        sharp_ratio = portfolio.sharpe_ratio()
        sortino_ratio = portfolio.sortino_ratio()
        max_drawdown = portfolio.max_drawdown()
        total_return = portfolio.total_return()
        
        print(f"Metrics for {regime}: Sharpe {sharp_ratio}, Sortino {sortino_ratio}, Max Drawdown {max_drawdown}, Total Return {total_return}")

        return {
            'Sharpe Ratio': sharp_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Total Return': total_return
        }

