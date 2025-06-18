import glob
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union, TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    from dataset_loader import DatasetLoader

def _discover_label_cols(columns: Iterable[str]) -> List[str]:
    """Find all columns that look like labels."""
    return [c for c in columns if c.lower() in ("label", "sentiment", "class")]

def _find_close_col(df: pd.DataFrame) -> str:
    """Find the column name for closing price."""
    close_cols = [c for c in df.columns if c.lower() in ("close", "closing", "price")]
    if not close_cols:
        raise ValueError("No closing price column found")
    return close_cols[0]

class DatasetLoader:
    def __init__(
        self,
        config_path_or_loader: Union[str, "DatasetLoader"] = "config.yaml",
        *,
        include_labels: bool = True,
        label_reducer: Optional[Callable[[pd.Series], Union[float, int, str]]] = None,
    ):
        print(f"DatasetLoader.__init__ called with: {config_path_or_loader}")
        
        if isinstance(config_path_or_loader, DatasetLoader):
            # Copy attributes from existing loader
            print("Copying attributes from existing DatasetLoader")
            self.prebit_dir = config_path_or_loader.prebit_dir
            self.price_path = config_path_or_loader.price_path
            self.kaggle_dir = config_path_or_loader.kaggle_dir
            self.events_path = config_path_or_loader.events_path
        else:
            # Load from config file
            print(f"Loading config from: {config_path_or_loader}")
            with open(config_path_or_loader, "r") as f:
                cfg = yaml.safe_load(f)
            d = cfg["data"]

            # PreBit directories / single file
            self.prebit_dir    = Path(d["prebit_dataset_dir"])
            self.price_path    = Path(d["price_label_path"])
            self.kaggle_dir    = Path(d["kaggle_dataset_dir"])
            self.events_path   = Path(d["bitcoin_events_path"])

        self.include_labels = include_labels
        self.label_reducer  = label_reducer or (lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        
        print(f"DatasetLoader initialized with paths:")
        print(f"  prebit_dir: {self.prebit_dir}")
        print(f"  price_path: {self.price_path}")
        print(f"  kaggle_dir: {self.kaggle_dir}")
        print(f"  events_path: {self.events_path}")

    def load_dataset(self, *, aggregate: bool = False) -> pd.DataFrame:
        # -- 1) Load PreBit tweets (2015-2021) --
        prebit_df = None
        if self.prebit_dir:
            prebit_df = self._load_prebit_dir()
            prebit_df = prebit_df.rename(columns={"Tweet Date": "date"})
            # Flag tweets from PreBit dataset
            prebit_df["data_source"] = "prebit"
            # Set metadata columns to NaN for PreBit tweets
            for col in ["hashtags", "is_retweet", "user_verified", "user_followers"]:
                prebit_df[col] = pd.NA

        # -- 2) Load Kaggle tweets (2021-2023) with rich metadata --
        kaggle_df = None
        if self.kaggle_dir:
            kaggle_df = self._load_kaggle_tweets()
            kaggle_df = kaggle_df.rename(columns={"Tweet Date": "date"})
            kaggle_df["data_source"] = "kaggle"
            # Ensure all metadata columns exist (even if not in source)
            for col in ["hashtags", "is_retweet", "user_verified", "user_followers"]:
                if col not in kaggle_df.columns:
                    kaggle_df[col] = pd.NA

        # -- 3) Combine all tweets --
        parts = []
        if prebit_df is not None:
            parts.append(prebit_df)
        if kaggle_df is not None:
            parts.append(kaggle_df)
        df = pd.concat(parts, ignore_index=True).sort_values("date")

        # -- 4) Merge price data with ALL tweets --
        price_df = self._load_price_data()
        df = pd.merge(df, price_df, on="date", how="left")

        # -- 5) Add metadata availability flags --
        df["has_user_metadata"] = df["user_followers"].notna()
        df["has_tweet_metadata"] = df["hashtags"].notna()

        # -- 6) Merge in events as a one-hot Is_Event flag --
        if self.events_path:
            events_df = self._load_events_data()
            df["Is_Event"] = df["date"].isin(events_df["date"]).astype(int)
        else:
            df["Is_Event"] = 0

        # -- 7) Per-day aggregation if requested --
        if aggregate:
            df = self._aggregate_per_day(df)

        return df

    def _load_kaggle_tweets(self) -> pd.DataFrame:
        """Load the additional Kaggle tweets CSV—must have date + text col."""
        df = pd.read_csv(Path(self.kaggle_file), parse_dates=["date","Date","Tweet Date"], dayfirst=False)

        # Find a text column
        text_cols = [c for c in df.columns if c.lower() in ("text","tweet","text_split")]
        if not text_cols:
            raise ValueError("No tweet/text column found in Kaggle CSV")
        text_col = text_cols[0]

        df = df.rename(columns={text_col:"Tweet Content", "date":"Tweet Date","Date":"Tweet Date"})
        # if labels present, keep them
        if not self.include_labels:
            df = df.drop(columns=_discover_label_cols(df.columns), errors="ignore")

        return df[["Tweet Date","Tweet Content"] + _discover_label_cols(df.columns)]

    # -- reuse existing _load_single_prebit, _load_prebit_dir, _aggregate_per_day --  

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_single_prebit(self) -> pd.DataFrame:
        """Load the (older) *single‑CSV* version of the PreBit dataset."""
        csv_path = Path(self.prebit_dir)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path, parse_dates=["date"])  # assumed column names
        required = {"date", "tweet", "close"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Expected at least columns {required}, got {set(df.columns)}"
            )

        if not self.include_labels:
            df = df.drop(columns=_discover_label_cols(df.columns), errors="ignore")

        return df.rename(columns={"date": "Tweet Date", "tweet": "Tweet Content", "close": "Close"})

    def _load_prebit_dir(self) -> pd.DataFrame:
        """Load the *directory* layout shipped on Kaggle (6 tweet CSVs + price)."""
        dir_path = Path(self.prebit_dataset_dir)
        if not dir_path.exists():
            raise FileNotFoundError(dir_path)

        # ---- 1. Tweets ---------------------------------------------------
        tweet_csvs = sorted(dir_path.glob("combined_tweets_*_labeled.csv"))
        if not tweet_csvs:
            raise FileNotFoundError("No tweet CSVs matching 'combined_tweets_*_labeled.csv' found in " + str(dir_path))

        tweet_frames: List[pd.DataFrame] = []
        for csv in tweet_csvs:
            df = pd.read_csv(csv, parse_dates=["date"])
            required = {"date", "text_split"}
            if not required.issubset(df.columns):
                raise ValueError(f"{csv} is missing one of {required}")

            keep_cols = ["date", "text_split"]
            if self.include_labels:
                label_cols = _discover_label_cols(df.columns)
                keep_cols.extend(label_cols)
            tweet_frames.append(df[keep_cols])

        tweets = pd.concat(tweet_frames, ignore_index=True)
        tweets = tweets.rename(columns={"date": "Tweet Date", "text_split": "Tweet Content"})

        # ---- 2. Price ----------------------------------------------------
        price_path = dir_path / "price_label.csv"
        if not price_path.exists():
            # price file optional – we simply return tweets only
            return tweets

        price_df = pd.read_csv(price_path, parse_dates=["date"])
        close_col = _find_close_col(price_df)

        price_keep = ["date", close_col]
        if self.include_labels:
            price_keep.extend(_discover_label_cols(price_df.columns))

        price_df = price_df[price_keep]
        price_df = price_df.rename(columns={"date": "Tweet Date", close_col: "Close"})

        # ---- 3. Merge ----------------------------------------------------
        # Many tweets → one price row per day ⇒ left join keeps all tweets.
        merged = tweets.merge(price_df, on="Tweet Date", how="left", suffixes=("", "_price"))
        return merged

    # ------------------------------------------------------------------
    # Aggregation helper
    # ------------------------------------------------------------------

    def _aggregate_per_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse multiple tweets into a single record per date."""
        label_cols = _discover_label_cols(df.columns) if self.include_labels else []

        numeric_cols = df.select_dtypes("number").columns.difference(label_cols)
        string_cols = df.select_dtypes("object").columns.difference(["Tweet Content"])

        def _agg_fn(series: pd.Series):
            if series.name in label_cols:
                return self.label_reducer(series)
            if series.name in numeric_cols:
                return series.mean()
            if series.name == "Tweet Content":
                return " \n".join(series.astype(str))
            # fallback
            return series.iloc[0]

        grouped = df.groupby("Tweet Date", as_index=False).agg(_agg_fn)
        return grouped

    def _load_price_data(self) -> pd.DataFrame:
        """
        Load daily close price data, renaming the time and close columns to 'date' and 'Close'.
        Returns a DataFrame with 'date', 'Close', and 'Volume' columns.
        """
        df = pd.read_csv(self.price_path, parse_dates=["Open time"])
        # Rename columns for consistency
        df = df.rename(columns={"Open time": "date", "Close": "Close", "Volume": "Volume"})
        # Keep date, Close and Volume columns
        return df[["date", "Close", "Volume"]]

    def _load_events_data(self) -> pd.DataFrame:
        """
        Load the events CSV and return a DataFrame with a 'date' column.
        """
        df = pd.read_csv(self.events_path, parse_dates=["Date", "date"])
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        return df[["date"]].drop_duplicates()

