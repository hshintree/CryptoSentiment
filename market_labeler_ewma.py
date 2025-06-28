"""market_labeler_ewma.py

Enhanced Triple‑Barrier labeler with EWMA volatility and proper financial formulas.

Fixed Triple‑Barrier labeler that properly handles intra-day data.

The key fix: When multiple tweets exist on the same day, all tweets on that day
should look at the SAME future price path (next unique trading days), not the
next rows in the dataset.
^^^^It should do this at least^^^
Key features:
- 30-day EWMA volatility σₜ from log-returns
- Volatility-adjusted barriers: Uₜ = Pₜ + Pₜ·σₜ·Fᵤ, Lₜ = Pₜ – Pₜ·σₜ·Fₗ
- 2-day minimum trend enforcement (label as Neutral if hit before day 2)
- 8-day ROC window with ±1σ thresholds
- Risk-adjusted Sharpe: (returns - 4% risk-free) / std * √252
- Maintains "next Vₜ trading days" logic for intra-day data
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Configuration dataclass ----------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class _TBLConfig:
    fu_grid: list[float]  # e.g. [0.5, 1.0, 1.5] (volatility multipliers)
    fl_grid: list[float]  # e.g. [0.5, 1.0, 1.5] (volatility multipliers)
    vt_grid: list[int]    # e.g. [3, 5, 7] (days)
    rebalance_days: int   # optimise every N obs (≈ 126 for 6 months)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "_TBLConfig":
        cfg = yaml.safe_load(open(path))
        mkt = cfg.get("market_labeling", {})
        return cls(
            fu_grid=mkt.get("fu_grid", [0.5, 1.0, 1.5]),
            fl_grid=mkt.get("fl_grid", [0.5, 1.0, 1.5]),
            vt_grid=mkt.get("vt_grid", [3, 5, 7]),
            rebalance_days=mkt.get("rebalance_days", 126),
        )

# ---------------------------------------------------------------------------
# Enhanced financial calculations ----------------------------------------
# ---------------------------------------------------------------------------

def _compute_ewma_volatility(prices: np.ndarray, tau: int = 30) -> np.ndarray:
    """
    Compute 30-day EWMA volatility from log-returns.
    
    Args:
        prices: Daily price series
        tau: EWMA decay parameter (30 days)
    
    Returns:
        EWMA volatility series (same length as prices, first values are NaN)
    """
    if len(prices) < 2:
        return np.full(len(prices), np.nan)
    
    # Compute log returns
    log_returns = np.diff(np.log(prices))
    
    # EWMA parameters
    alpha = 2.0 / (tau + 1)  # Decay factor
    
    # Initialize EWMA volatility array
    ewma_vol = np.full(len(prices), np.nan)
    
    if len(log_returns) == 0:
        return ewma_vol
    
    # Initialize with first return's squared value
    ewma_var = log_returns[0] ** 2
    ewma_vol[1] = np.sqrt(ewma_var)
    
    # Compute EWMA variance recursively
    for i in range(1, len(log_returns)):
        ewma_var = alpha * (log_returns[i] ** 2) + (1 - alpha) * ewma_var
        ewma_vol[i + 1] = np.sqrt(ewma_var)
    
    return ewma_vol

def _compute_roc_thresholds(prices: np.ndarray, window: int = 8) -> tuple[float, float]:
    """
    Compute ROC over 8-day window and set thresholds to ±1σ.
    
    Args:
        prices: Daily price series
        window: ROC window length
        
    Returns:
        (lower_threshold, upper_threshold) as ±1σ of ROC series
    """
    if len(prices) < window + 1:
        return -0.05, 0.05  # Default 5% thresholds
    
    # Compute ROC over window
    roc_series = []
    for i in range(window, len(prices)):
        roc = (prices[i] - prices[i - window]) / prices[i - window]
        roc_series.append(roc)
    
    if len(roc_series) == 0:
        return -0.05, 0.05
    
    roc_std = np.std(roc_series, ddof=1)
    return -roc_std, roc_std

def _risk_adjusted_sharpe(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Compute risk-adjusted Sharpe ratio.
    
    Args:
        returns: Daily return series
        risk_free_rate: Annual risk-free rate (default 4%)
        
    Returns:
        Sharpe ratio: (mean_return - risk_free) / std * √252
    """
    if len(returns) == 0 or returns.std(ddof=1) == 0:
        return -np.inf
    
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / 252
    
    # Compute excess returns
    excess_returns = returns - daily_rf
    
    # Sharpe ratio with annualization
    return excess_returns.mean() / returns.std(ddof=1) * np.sqrt(252)

# ---------------------------------------------------------------------------
# Enhanced Main labeler ------------------------------------------------------
# ---------------------------------------------------------------------------

class MarketLabelerTBL:
    def __init__(self, cfg_path: str = "config.yaml") -> None:
        self.cfg = _TBLConfig.from_yaml(cfg_path)
        # Store fitted thresholds to prevent leakage
        self._fitted_thresholds = None
        self._fitted_volatility_params = None

    # ------------------------------------------------------------------
    def label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced TBL labels with EWMA volatility and financial constraints.

        Parameters
        ----------
        data : DataFrame with at least columns ``date`` & ``Close``.
        """
        if "date" not in data.columns and "Tweet Date" in data.columns:
            data = data.rename(columns={"Tweet Date": "date"})
        df = data.copy().sort_values("date").reset_index(drop=True)

        # Create unique daily price data for labeling logic
        print("Creating unique daily price series for EWMA-based TBL labeling...")
        daily_prices = self._create_daily_price_series(df)
        
        # Compute EWMA volatility and ROC thresholds
        print("Computing EWMA volatility and ROC thresholds...")
        daily_prices = self._add_financial_indicators(daily_prices)
        
        # Pre-allocate arrays for all tweets
        n = len(df)
        upper = np.empty(n)
        lower = np.empty(n) 
        label = np.empty(n, dtype=object)
        vertical_barrier = np.empty(n)
        barrier_end_date = np.empty(n, dtype="datetime64[ns]")
        volatility = np.empty(n)

        # Rolling optimization on daily price data
        daily_step = self.cfg.rebalance_days
        
        for daily_start in range(0, len(daily_prices), daily_step):
            daily_end = min(daily_start + daily_step, len(daily_prices))
            
            # Grid-search on the *previous* daily window
            opt_slice = slice(max(0, daily_start - daily_step), daily_start)
            if opt_slice.start < opt_slice.stop:
                fu, fl, vt = self._optimize_params(daily_prices.iloc[opt_slice])
            else:
                # First window - use default params
                fu, fl, vt = self.cfg.fu_grid[0], self.cfg.fl_grid[0], self.cfg.vt_grid[0]
            
            # Apply enhanced TBL to all tweets in this daily window
            daily_window = daily_prices.iloc[daily_start:daily_end]
            self._apply_enhanced_tbl_to_window(
                df, daily_window, fu, fl, vt, 
                upper, lower, label, vertical_barrier, barrier_end_date, volatility
            )

        df["Upper Barrier"] = upper
        df["Lower Barrier"] = lower
        df["Vertical Barrier"] = vertical_barrier
        df["Barrier End Date"] = pd.to_datetime(barrier_end_date)
        df["Volatility"] = volatility
        df["Label"] = label
        
        return df

    # ------------------------------------------------------------------
    def fit_and_label(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit labeler on training data and apply labels.
        Computes thresholds from training data only to prevent leakage.
        """
        print("Fitting EWMA labeler on training data to prevent leakage...")
        
        # First, fit the thresholds on training data
        self._fit_thresholds(train_data)
        
        # Then apply labels using fitted thresholds
        return self._apply_labels_with_fitted_thresholds(train_data)
    
    def apply_labels(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply labels to test/validation data using previously fitted thresholds.
        """
        if self._fitted_thresholds is None:
            raise ValueError("Must call fit_and_label() on training data first!")
        
        print("Applying fitted thresholds to validation data...")
        return self._apply_labels_with_fitted_thresholds(test_data)
    
    def _fit_thresholds(self, data: pd.DataFrame) -> None:
        """
        Fit ROC thresholds and volatility parameters on training data only.
        NO GLOBAL COMPUTATION - will compute per-time-step to prevent leakage.
        """
        # Standardize date column name for internal processing
        original_date_col = None
        if "Tweet Date" in data.columns and "date" not in data.columns:
            original_date_col = "Tweet Date"
            data = data.rename(columns={"Tweet Date": "date"})
        df = data.copy().sort_values("date").reset_index(drop=True)
        
        # Create daily price series from training data only
        daily_prices = self._create_daily_price_series(df)
        
        # Store daily prices for per-time-step computation
        self._fitted_thresholds = {
            "daily_prices": daily_prices["Close"].values,
            "fit_mode": "per_timestep"  # Flag to use per-timestep computation
        }
        
        print(f"Stored {len(daily_prices)} training days for per-timestep threshold computation")
        print("ROC/EWMA thresholds will be computed per-timestep to prevent leakage")
    
    def _apply_labels_with_fitted_thresholds(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply labels using pre-fitted thresholds (no peeking at future data).
        """
        # Standardize date column name for internal processing
        original_date_col = None
        if "Tweet Date" in data.columns and "date" not in data.columns:
            original_date_col = "Tweet Date"
            data = data.rename(columns={"Tweet Date": "date"})
        df = data.copy().sort_values("date").reset_index(drop=True)
        
        # Create daily price series
        daily_prices = self._create_daily_price_series(df)
        
        # Add per-timestep EWMA volatility computation (no look-ahead)
        train_prices = self._fitted_thresholds["daily_prices"]
        daily_prices_vals = daily_prices["Close"].values
        
        # Compute per-timestep volatility and thresholds
        ewma_vols = []
        roc_lowers = []
        roc_uppers = []
        
        for i in range(len(daily_prices)):
            # For training: use prices[:i+1] from training data
            # For validation: use training prices + validation prices[:i+1]
            if len(train_prices) > 0:
                # Use training prices + current validation history
                available_prices = np.concatenate([train_prices, daily_prices_vals[:i+1]])
            else:
                # Fallback if no training prices
                available_prices = daily_prices_vals[:i+1]
            
            # Compute EWMA volatility using only available history
            if len(available_prices) >= 2:
                vol_series = _compute_ewma_volatility(available_prices, tau=30)
                current_vol = vol_series[-1] if not np.isnan(vol_series[-1]) else 0.05
            else:
                current_vol = 0.05
                
            # Compute ROC thresholds using only available history  
            if len(available_prices) >= 9:  # Need at least window + 1
                roc_lower, roc_upper = _compute_roc_thresholds(available_prices, window=8)
            else:
                roc_lower, roc_upper = -0.05, 0.05  # Default 5%
                
            ewma_vols.append(current_vol)
            roc_lowers.append(roc_lower)
            roc_uppers.append(roc_upper)
        
        daily_prices["EWMA_Volatility"] = ewma_vols
        daily_prices["ROC_Lower"] = roc_lowers  
        daily_prices["ROC_Upper"] = roc_uppers
        
        # Apply TBL logic with per-timestep parameters
        n = len(df)
        upper = np.empty(n)
        lower = np.empty(n) 
        label = np.empty(n, dtype=object)
        vertical_barrier = np.empty(n)
        barrier_end_date = np.empty(n, dtype="datetime64[ns]")
        volatility = np.empty(n)
        
        # Use default parameters for simplicity in fold-wise training
        fu, fl, vt = self.cfg.fu_grid[0], self.cfg.fl_grid[0], self.cfg.vt_grid[0]
        
        # Apply TBL to all tweets
        self._apply_enhanced_tbl_to_window(
            df, daily_prices, fu, fl, vt, 
            upper, lower, label, vertical_barrier, barrier_end_date, volatility
        )
        
        df["Upper Barrier"] = upper
        df["Lower Barrier"] = lower
        df["Vertical Barrier"] = vertical_barrier
        df["Barrier End Date"] = pd.to_datetime(barrier_end_date)
        df["Volatility"] = volatility
        df["Label"] = label
        
        # Restore original date column name if it was changed
        if original_date_col is not None:
            df = df.rename(columns={"date": original_date_col})
        
        return df

    def _create_daily_price_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a series of unique daily prices for TBL logic."""
        # Group by date (date only) and take first price of each day
        df_daily = df.copy()
        df_daily['date_only'] = df_daily['date'].dt.date
        
        daily_prices = df_daily.groupby('date_only').agg({
            'date': 'first',
            'Close': 'first'
        }).reset_index(drop=True)
        
        daily_prices = daily_prices.sort_values('date').reset_index(drop=True)
        print(f"Created daily price series: {len(daily_prices)} unique trading days")
        
        return daily_prices

    def _add_financial_indicators(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """Add EWMA volatility and other financial indicators."""
        prices = daily_prices['Close'].values
        
        # Compute 30-day EWMA volatility
        volatility = _compute_ewma_volatility(prices, tau=30)
        daily_prices['EWMA_Volatility'] = volatility
        
        # Compute ROC thresholds (for future use if needed)
        roc_lower, roc_upper = _compute_roc_thresholds(prices, window=8)
        daily_prices['ROC_Lower_Threshold'] = roc_lower
        daily_prices['ROC_Upper_Threshold'] = roc_upper
        
        print(f"EWMA volatility computed. Mean volatility: {np.nanmean(volatility):.4f}")
        print(f"ROC thresholds: [{roc_lower:.4f}, {roc_upper:.4f}]")
        
        return daily_prices

    def _apply_enhanced_tbl_to_window(
        self,
        df: pd.DataFrame,
        daily_window: pd.DataFrame,
        fu: float,
        fl: float,
        vt: int,
        upper: np.ndarray,
        lower: np.ndarray,
        label: np.ndarray,
        vertical_barrier: np.ndarray,
        barrier_end_date: np.ndarray,
        volatility: np.ndarray
    ):
        """Apply enhanced TBL with volatility-adjusted barriers and trend constraints."""

        # 1) reset the daily slice so that .iloc is purely positional (0…n-1)
        window_prices = daily_window.reset_index(drop=True)

        # 2) build a date→position map off of that reset frame
        date_to_pos = {
            row["date"].date(): pos
            for pos, row in window_prices.iterrows()
        }

        # 3) find which tweets fall into this block of trading days
        start_date = daily_window["date"].min().date()
        end_date   = daily_window["date"].max().date()
        window_mask   = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        window_tweets = df[window_mask]

        for tweet_idx in window_tweets.index:
            tweet_date  = df.at[tweet_idx, "date"].date()
            tweet_price = df.at[tweet_idx, "Close"]

            if tweet_date not in date_to_pos:
                # edge case: no matching trading‐day row
                upper[tweet_idx]           = tweet_price * 1.02
                lower[tweet_idx]           = tweet_price * 0.98
                vertical_barrier[tweet_idx] = vt
                barrier_end_date[tweet_idx] = df.at[tweet_idx, "date"] + pd.Timedelta(days=vt)
                volatility[tweet_idx]      = 0.02
                label[tweet_idx]           = "Neutral"
                continue

            # positional index into window_prices
            daily_idx = date_to_pos[tweet_date]

            # pull that day's EWMA volatility
            ewma_vol = window_prices.loc[daily_idx, "EWMA_Volatility"]
            if pd.isna(ewma_vol) or ewma_vol <= 0:
                ewma_vol = 0.02  # safe default

            # compute the vol-adjusted barriers
            upper[tweet_idx]           = tweet_price + tweet_price * ewma_vol * fu
            lower[tweet_idx]           = tweet_price - tweet_price * ewma_vol * fl
            vertical_barrier[tweet_idx] = vt
            barrier_end_date[tweet_idx] = df.at[tweet_idx, "date"] + pd.Timedelta(days=vt)
            volatility[tweet_idx]      = ewma_vol

            # ── examine the future path (≤ vt trading days) ───────────────
            future_end    = min(daily_idx + vt, len(window_prices) - 1)
            future_prices = window_prices["Close"].iloc[daily_idx : future_end + 1].values

            label[tweet_idx], offset = self._enhanced_tbl_logic_with_offset(
                future_prices,
                upper[tweet_idx],
                lower[tweet_idx],
                min_days=2
            )
            # store when that regime actually finished
            end_pos = min(daily_idx + offset, len(window_prices) - 1)
            barrier_end_date[tweet_idx] = window_prices.at[end_pos, "date"].normalize()


    def _enhanced_tbl_logic(self, future_prices: np.ndarray, upper_barrier: float, 
                          lower_barrier: float, min_days: int = 2) -> str:
        """
        Enhanced TBL logic with 2-day minimum trend enforcement.
        
        Args:
            future_prices: Price path starting from current day
            upper_barrier: Upper barrier level
            lower_barrier: Lower barrier level
            min_days: Minimum days before allowing barrier hits (default 2)
            
        Returns:
            Label: "Bullish", "Bearish", or "Neutral"
        """
        if len(future_prices) < min_days + 1:
            return "Neutral"
        
        # Only consider prices from min_days onwards for barrier hits
        relevant_prices = future_prices[min_days:]
        
        if len(relevant_prices) == 0:
            return "Neutral"
        
        # Check barrier hits
        hit_upper = np.any(relevant_prices >= upper_barrier)
        hit_lower = np.any(relevant_prices <= lower_barrier)
        
        if hit_upper and hit_lower:
            # Whichever comes first wins (among valid days)
            up_idx = np.argmax(relevant_prices >= upper_barrier)
            lo_idx = np.argmax(relevant_prices <= lower_barrier)
            return "Bullish" if up_idx < lo_idx else "Bearish"
        elif hit_upper:
            return "Bullish"
        elif hit_lower:
            return "Bearish"
        else:
            return "Neutral"

    def _enhanced_tbl_logic_with_offset(self, future_prices: np.ndarray,
                                        upper_barrier: float,
                                        lower_barrier: float,
                                        min_days: int = 2) -> tuple[str,int]:
        """
        Enhanced TBL logic with 2-day minimum trend enforcement and offset tracking.
        
        Args:
            future_prices: Price path starting from current day
            upper_barrier: Upper barrier level
            lower_barrier: Lower barrier level
            min_days: Minimum days before allowing barrier hits (default 2)
            
        Returns:
            Label: "Bullish", "Bearish", or "Neutral"
            Offset: Number of days until regime ends
        """
        if len(future_prices) < min_days + 1:
            return "Neutral", len(future_prices)-1
        
        # Only consider prices from min_days onwards for barrier hits
        relevant_prices = future_prices[min_days:]
        
        if len(relevant_prices) == 0:
            return "Neutral", len(future_prices)-1
        
        # Check barrier hits
        hit_upper = np.any(relevant_prices >= upper_barrier)
        hit_lower = np.any(relevant_prices <= lower_barrier)
        
        if hit_upper and hit_lower:
            up_idx = np.argmax(relevant_prices >= upper_barrier)
            lo_idx = np.argmax(relevant_prices <= lower_barrier)
            lbl    = "Bullish" if up_idx < lo_idx else "Bearish"
            off    = min(up_idx, lo_idx) + min_days
        elif hit_upper:
            lbl, off = "Bullish", np.argmax(relevant_prices >= upper_barrier) + min_days
        elif hit_lower:
            lbl, off = "Bearish", np.argmax(relevant_prices <= lower_barrier) + min_days
        else:
            lbl, off = "Neutral", len(future_prices)-1
        return lbl, off

    # ------------------------------------------------------------------
    def _optimize_params(self, daily_window: pd.DataFrame) -> tuple[float, float, int]:
        """Grid‑search Fu, Fl, Vt on daily price data to max risk-adjusted Sharpe."""
        if len(daily_window) < 3:
            return self.cfg.fu_grid[0], self.cfg.fl_grid[0], self.cfg.vt_grid[0]
        
        # Ensure we have volatility data
        if 'EWMA_Volatility' not in daily_window.columns:
            daily_window = self._add_financial_indicators(daily_window)
            
        best = (-np.inf, self.cfg.fu_grid[0], self.cfg.fl_grid[0], self.cfg.vt_grid[0])
        
        for fu, fl, vt in itertools.product(self.cfg.fu_grid, self.cfg.fl_grid, self.cfg.vt_grid):
            labels = self._apply_enhanced_tbl_daily(daily_window, fu, fl, vt)
            rets = self._strategy_returns(daily_window['Close'].values, labels)
            s = _risk_adjusted_sharpe(pd.Series(rets))
            if s > best[0]:
                best = (s, fu, fl, vt)
        
        return best[1:]

    # ------------------------------------------------------------------
    def _apply_enhanced_tbl_daily(self, daily_window: pd.DataFrame, 
                                fu: float, fl: float, vt: int) -> np.ndarray:
        """Apply enhanced TBL to daily price series for optimization."""
        prices = daily_window['Close'].values
        volatilities = daily_window['EWMA_Volatility'].values
        n = len(prices)
        labels = np.empty(n, dtype=object)

        for i in range(n):
            # Get volatility for this day
            vol = volatilities[i] if not pd.isna(volatilities[i]) else 0.02
            
            # Compute volatility-adjusted barriers
            upper_barrier = prices[i] + prices[i] * vol * fu
            lower_barrier = prices[i] - prices[i] * vol * fl
            
            # Future price path
            future_end = min(i + vt, n - 1)
            future_prices = prices[i:future_end + 1]
            
            # Apply enhanced logic
            labels[i] = self._enhanced_tbl_logic(future_prices, upper_barrier, lower_barrier, min_days=2)
        
        return labels

    # ------------------------------------------------------------------
    @staticmethod
    def _strategy_returns(prices: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Generate daily returns for risk-adjusted Sharpe optimization."""
        if len(prices) < 2:
            return np.array([])
        pos = np.where(labels == "Bullish", +1, np.where(labels == "Bearish", -1, 0))
        pct = np.diff(prices) / prices[:-1]
        return pos[:-1] * pct


# Test function to validate the enhanced labeler
# def test_enhanced_labeler():
#     """Test the enhanced labeler with EWMA volatility."""
#     print("="*60)
#     print("TESTING ENHANCED MARKET LABELER WITH EWMA")
#     print("="*60)
    
#     # Load sample data
#     df = pd.read_csv("data/combined_dataset_raw.csv")
#     df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
#     df = df.dropna(subset=['date'])
#     df = df[df['Close'].notna()]
    
#     # Take a sample for testing
#     start_date = '2020-01-01'
#     end_date = '2020-01-31'
#     df_sample = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
#     print(f"Test sample: {len(df_sample)} tweets from {start_date} to {end_date}")
#     print(f"Unique dates: {df_sample['date'].dt.date.nunique()}")
    
#     # Test enhanced labeler
#     labeler = MarketLabelerEWMA("config.yaml")
#     result = labeler.label_data(df_sample)
    
#     print("\nEnhanced labeling results:")
#     label_counts = result['Label'].value_counts()
#     print(label_counts)
#     print(f"Percentages: {(label_counts / len(result) * 100).round(2)}")
    
#     print(f"\nVolatility statistics:")
#     print(f"Mean volatility: {result['Volatility'].mean():.4f}")
#     print(f"Volatility range: [{result['Volatility'].min():.4f}, {result['Volatility'].max():.4f}]")
    
#     # Show sample results
#     print("\nSample results with volatility-adjusted barriers:")
#     sample_cols = ['date', 'Close', 'Volatility', 'Upper Barrier', 'Lower Barrier', 'Label']
#     print(result[sample_cols].head(10))


# if __name__ == "__main__":
#     test_enhanced_labeler() 


class MarketFeatureGenerator:
    """
    Causal "previous-regime" feature.

    For every tweet we return the label of the **last completed regime**,
    i.e. the most recent window whose barrier had already closed strictly
    before the tweet's date.  Works fold-locally, needs no global state and
    never peeks into the future → leakage-free.
    """
    def fit(self, *_):                 # nothing to learn – stateless
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if "Barrier End Date" not in df.columns:
            raise ValueError("Run MarketLabelerTBL before generating features")

        ordered = (
            df[["Tweet Date", "Barrier End Date", "Label"]]
              .sort_values("Tweet Date")
              .reset_index()
        )

        # collect all barrier-closing events first
        events = (
            ordered[["Barrier End Date", "Label"]]
              .dropna()
              .drop_duplicates()
              .sort_values("Barrier End Date")
              .itertuples(index=False, name=None)
        )
        events = list(events)
        ev_ptr = 0
        last_lbl = "Neutral"
        prev_labels = []

        for _, row in ordered.iterrows():
            today = row["Tweet Date"].normalize()
            while ev_ptr < len(events) and events[ev_ptr][0] < today:
                last_lbl = events[ev_ptr][1]
                ev_ptr  += 1
            prev_labels.append(last_lbl)

        out = pd.Series(prev_labels, index=ordered["index"])
        return out.reindex(df.index).fillna("Neutral")