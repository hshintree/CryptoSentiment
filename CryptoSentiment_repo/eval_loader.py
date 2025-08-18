from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import TimeSeriesSplit
import yaml
import glob
import itertools

class EvalLoader:
    """
    Self-contained loader for creating evaluation datasets Ea and Eb.
    
    Ea: In-sample 2020 dataset with ~60k balanced tweets
    Eb: Out-of-sample 2015-2023 event-focused dataset with ~40k tweets
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with config path and load necessary paths."""
        print(f"EvalLoader.__init__ called with: {config_path}")
        
        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # Get data paths from config
        data_cfg = cfg.get("data", {})
        self.prebit_dir = Path(data_cfg.get("prebit_dataset_dir"))
        self.price_path = Path(data_cfg.get("price_label_path"))
        self.kaggle_dir = Path(data_cfg.get("kaggle_dataset_dir"))
        self.events_path = Path(data_cfg.get("bitcoin_events_path"))
        
        # Market labeling config
        mkt_cfg = cfg.get("market_labeling", {})
        self.fu_grid = mkt_cfg.get("fu_grid", [0.04, 0.06, 0.08])
        self.fl_grid = mkt_cfg.get("fl_grid", [0.04, 0.06, 0.08])
        self.vt_grid = mkt_cfg.get("vt_grid", [5, 10])
        self.rebalance_days = mkt_cfg.get("rebalance_days", 126)
        
        print("EvalLoader initialization complete")
    
    def create_eval_datasets(self, n_folds: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Create both Ea and Eb evaluation datasets with their respective folds.
        Returns a dictionary containing:
        - ea_train_folds: list of DataFrames for each Ea training fold
        - ea_test_folds: list of DataFrames for each Ea testing fold
        - eb_train_folds: list of DataFrames for each Eb training fold
        - eb_test_folds: list of DataFrames for each Eb testing fold
        - ea_full: complete Ea dataset
        - eb_full: complete Eb dataset
        - all_data: merged daily-aggregated dataset
        """
        # Load and preprocess all data
        df = self._load_dataset()
        
        # Add market-derived labels using triple-barrier method
        df = self._label_data(df)
        
        # Create Ea (2020 in-sample)
        ea_df = self._create_ea_dataset(df)
        ea_train_folds, ea_test_folds = self._create_time_folds(ea_df, n_folds)
        
        # Create Eb (2015-2023 event-focused, excluding 2020)
        eb_df = self._create_eb_dataset(df)
        eb_train_folds, eb_test_folds = self._create_time_folds(eb_df, n_folds)
        
        # Create aggregated version
        all_data = self._aggregate_all_data(pd.concat([ea_df, eb_df]))
        
        return {
            'ea_train_folds': ea_train_folds,
            'ea_test_folds': ea_test_folds,
            'eb_train_folds': eb_train_folds,
            'eb_test_folds': eb_test_folds,
            'ea_full': ea_df,
            'eb_full': eb_df,
            'all_data': all_data
        }
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load and combine all data sources."""
        # Load PreBit tweets (2015-2021)
        prebit_df = self._load_prebit_data()
        
        # Load Kaggle tweets (2021-2023)
        kaggle_df = self._load_kaggle_data()
        
        # Load price data
        price_df = self._load_price_data()
        
        # Load events data
        events_df = self._load_events_data()
        
        # Combine all data
        df = pd.concat([prebit_df, kaggle_df])
        df = df.merge(price_df, on='date', how='left')
        df = df.merge(events_df, on='date', how='left')
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        return df
    
    def _load_prebit_data(self) -> pd.DataFrame:
        """Load PreBit dataset (2015-2021)."""
        tweet_csvs = sorted(self.prebit_dir.glob("combined_tweets_*_labeled.csv"))
        if not tweet_csvs:
            raise FileNotFoundError("No tweet CSVs found in PreBit directory")
        
        dfs = []
        for csv in tweet_csvs:
            df = pd.read_csv(csv, parse_dates=["date"])
            df = df.rename(columns={"date": "date", "text_split": "Tweet Content"})
            dfs.append(df[["date", "Tweet Content"]])
        
        return pd.concat(dfs, ignore_index=True)
    
    def _load_kaggle_data(self) -> pd.DataFrame:
        """Load Kaggle dataset (2021-2023)."""
        # Look for CSV files in the Kaggle directory
        kaggle_files = list(self.kaggle_dir.glob("*.csv"))
        if not kaggle_files:
            raise FileNotFoundError(f"No CSV files found in Kaggle directory: {self.kaggle_dir}")
        
        # Load and combine all CSV files
        dfs = []
        for csv_file in kaggle_files:
            # Read CSV with low_memory=False to handle mixed types
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Find the date column
            date_cols = [c for c in df.columns if c.lower() in ("date", "tweet date")]
            if not date_cols:
                raise ValueError(f"No date column found in {csv_file}")
            date_col = date_cols[0]
            
            # Find the text column
            text_cols = [c for c in df.columns if c.lower() in ("text", "tweet", "text_split")]
            if not text_cols:
                raise ValueError(f"No tweet/text column found in {csv_file}")
            text_col = text_cols[0]
            
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Standardize column names and keep only needed columns
            df = df.rename(columns={
                text_col: "Tweet Content",
                date_col: "date"
            })
            
            # Keep only needed columns and convert to datetime
            df = df[["date", "Tweet Content"]].copy()
            
            # Clean up memory
            dfs.append(df)
            del df
        
        # Combine all dataframes
        result = pd.concat(dfs, ignore_index=True)
        del dfs  # Clean up the list of dataframes
        
        return result
    
    def _load_price_data(self) -> pd.DataFrame:
        """Load price data with technical indicators."""
        df = pd.read_csv(self.price_path, parse_dates=["date"])
        return df[["date", "Close", "Volume"]]
    
    def _load_events_data(self) -> pd.DataFrame:
        """Load events data."""
        df = pd.read_csv(self.events_path, parse_dates=["date"])
        df["Is_Event"] = 1
        return df[["date", "Is_Event"]]
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and ROC technical indicators."""
        # Calculate RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Calculate ROC
        df["ROC"] = df["Close"].pct_change(periods=10) * 100
        
        # Add log volume
        df["log_volume"] = np.log1p(df["Volume"])
        
        return df
    
    def _label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply triple-barrier labeling to the data."""
        df = df.sort_values("date").reset_index(drop=True)
        prices = df["Close"].values
        
        # Pre-allocate arrays
        upper = np.empty(len(df))
        lower = np.empty(len(df))
        label = np.empty(len(df), dtype=object)
        vertical_barrier = np.empty(len(df))
        
        # Rolling optimization
        n = len(df)
        step = self.rebalance_days
        for start in range(0, n, step):
            end = min(start + step, n)
            win_prices = prices[start:end]
            
            # Grid-search on previous window
            opt_slice = slice(max(0, start - step), start)
            fu, fl, vt = self._optimize_params(prices[opt_slice])
            
            # Apply TBL with optimal params
            u, l, lab = self._apply_tbl(win_prices, fu, fl, vt)
            upper[start:end] = u
            lower[start:end] = l
            label[start:end] = lab
            vertical_barrier[start:end] = vt
        
        df["Upper Barrier"] = upper
        df["Lower Barrier"] = lower
        df["Vertical Barrier"] = vertical_barrier
        df["Label"] = label
        return df
    
    def _optimize_params(self, prices: np.ndarray) -> Tuple[float, float, int]:
        """Grid-search for optimal triple-barrier parameters."""
        best = (-np.inf, self.fu_grid[0], self.fl_grid[0], self.vt_grid[0])
        for fu, fl, vt in itertools.product(self.fu_grid, self.fl_grid, self.vt_grid):
            _, _, lab = self._apply_tbl(prices, fu, fl, vt)
            rets = self._strategy_returns(prices, lab)
            s = self._sharpe(pd.Series(rets))
            if s > best[0]:
                best = (s, fu, fl, vt)
        return best[1:]
    
    def _apply_tbl(self, prices: np.ndarray, fu: float, fl: float, vt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply triple-barrier labeling to price series."""
        n = len(prices)
        upper = prices * (1 + fu)
        lower = prices * (1 - fl)
        label = np.empty(n, dtype=object)
        
        for i in range(n):
            horizon = min(i + vt, n - 1)
            path = prices[i:horizon + 1]
            hit_upper = np.any(path >= upper[i])
            hit_lower = np.any(path <= lower[i])
            
            if hit_upper and hit_lower:
                up_idx = np.argmax(path >= upper[i])
                lo_idx = np.argmax(path <= lower[i])
                label[i] = "Bullish" if up_idx < lo_idx else "Bearish"
            elif hit_upper:
                label[i] = "Bullish"
            elif hit_lower:
                label[i] = "Bearish"
            else:
                label[i] = "Neutral"
        
        return upper[:n], lower[:n], label
    
    def _strategy_returns(self, prices: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate strategy returns for Sharpe ratio optimization."""
        pos = np.where(labels == "Bullish", +1, np.where(labels == "Bearish", -1, 0))
        pct = np.diff(prices) / prices[:-1]
        return pos[:-1] * pct
    
    def _sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return -np.inf
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _create_ea_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the Ea (2020 in-sample) dataset with balanced labels."""
        # Filter for 2020
        mask = df['date'].dt.year == 2020
        ea_df = df[mask].copy()
        
        # Sample ~60k tweets evenly across labels
        target_per_class = 20000  # 60k total
        balanced_samples = []
        for label in ['Bullish', 'Bearish', 'Neutral']:
            class_df = ea_df[ea_df['Label'] == label]
            if len(class_df) > target_per_class:
                class_df = class_df.sample(n=target_per_class, random_state=42)
            balanced_samples.append(class_df)
        
        ea_df = pd.concat(balanced_samples)
        ea_df = self._add_previous_day_label(ea_df)
        return ea_df.sort_values('date')
    
    def _create_eb_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the Eb (event-focused) dataset excluding 2020."""
        # Exclude 2020
        mask = df['date'].dt.year != 2020
        eb_df = df[mask].copy()
        
        # Split into event and non-event days
        event_tweets = eb_df[eb_df['Is_Event'] == 1]
        non_event_tweets = eb_df[eb_df['Is_Event'] == 0]
        
        # Sample from event days (prioritize these)
        event_sample = event_tweets.sample(n=min(25000, len(event_tweets)), random_state=42)
        
        # Sample from non-event days to balance
        remaining_samples = 40000 - len(event_sample)
        non_event_sample = non_event_tweets.sample(n=remaining_samples, random_state=42)
        
        # Combine and add previous day label
        eb_df = pd.concat([event_sample, non_event_sample])
        eb_df = self._add_previous_day_label(eb_df)
        return eb_df.sort_values('date')
    
    def _create_time_folds(self, df: pd.DataFrame, n_folds: int) -> Tuple[list, list]:
        """Create time-series cross-validation folds preserving date order."""
        tscv = TimeSeriesSplit(n_splits=n_folds)
        train_folds, test_folds = [], []
        
        for train_idx, test_idx in tscv.split(df):
            train_folds.append(df.iloc[train_idx])
            test_folds.append(df.iloc[test_idx])
            
        return train_folds, test_folds
    
    def _add_previous_day_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add previous day's label for each tweet."""
        df = df.sort_values('date')
        df['Previous_Label'] = df.groupby(df['date'].dt.date)['Label'].transform(
            lambda x: x.shift(1).fillna(method='ffill')
        )
        # Fill first day's labels with 'Neutral'
        df['Previous_Label'] = df['Previous_Label'].fillna('Neutral')
        return df
    
    def _aggregate_all_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tweets by day with majority label and mean numeric features."""
        # Numeric columns to average
        numeric_cols = ['Close', 'RSI', 'ROC', 'log_volume']
        
        # Group by date
        grouped = df.groupby(df['date'].dt.date).agg({
            'Label': lambda x: x.mode().iloc[0],  # majority label
            'Tweet Content': lambda x: ' \n'.join(x),  # concatenate tweets
            'Is_Event': 'max',  # if any tweet that day was an event
            **{col: 'mean' for col in numeric_cols}  # average numeric features
        }).reset_index()
        
        # Add previous day label
        grouped = self._add_previous_day_label(grouped)
        
        return grouped 