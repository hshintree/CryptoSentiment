"""dataset_loader.py

Enhanced dataset loader to handle the PreBit Bitcoin multimodal dataset
distributed on Kaggle (https://www.kaggle.com/datasets/zyz5557585/prebit-multimodal-dataset-for-bitcoin-price).

Key improvements over the previous version:

1. **Directory‑based loading** – Seamlessly loads yearly tweet CSVs that follow
   the pattern `combined_tweets_<YEAR>_labeled.csv` *and* the accompanying
   `price_label.csv`.
2. **Automatic label discovery** – Keeps *all* `label_*` columns and merges the
   tweet‑level labels with the price labels (suffixing clashes where needed).
3. **Config‑driven** – No hard‑coded paths.  Use `config.yaml → data →
   prebit_dataset_dir` (preferred) or set `prebit_dataset_path` for the single‑csv
   variant.
4. **Flexible aggregation** – Allows the caller to keep the raw many‑tweets‑per‑day
   format **or** aggregate to a single row per date with a custom reducer.
5. **Robustness** –  Clear validation of required columns and helpful error
   messages.

Example `config.yaml` snippet:

```yaml
# config.yaml
 data:
   prebit_dataset_dir: "/Users/hakeemshindy/Downloads/archive"
   preprocessing_steps:
     text_normalization: true
     remove_urls: true
 market_labeling:
   strategy: "TBL"
``` 

Usage:

```python
from dataset_loader import DatasetLoader

d loader = DatasetLoader("config.yaml")
# Keep raw tweets
raw = loader.load_dataset()

# One‑row‑per‑day aggregation and simple majority vote on labels
agg = loader.load_dataset(aggregate=True)
```
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_close_col(df: pd.DataFrame) -> str:
    """Return the name of the column that stores the daily *close* price."""
    for candidate in ("Close_x", "close", "Close"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not identify close price column (expected one of Close_x/close/Close)")


def _discover_label_cols(cols: Iterable[str]) -> List[str]:
    """Return all column names that start with the Kaggle ‑style label prefix."""
    return [c for c in cols if c.startswith("label_")]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Load PreBit multimodal dataset (tweets + price + labels).

    Parameters
    ----------
    config_path : str, default ``"config.yaml"``
        Path to a YAML config file containing at least the **data** section.
    include_labels : bool, default ``True``
        If *False*, drop the ``label_*`` columns entirely.
    label_reducer : Optional[Callable[[pd.Series], float|int|str]]
        If ``aggregate`` (see :py:meth:`load_dataset`) is ``True`` *and*
        ``include_labels`` is ``True``, this reducer is applied to every label
        column to go from many tweets per day → a single value per day.  By
        default a simple *mode* is used (majority vote).
    """

    def __init__(
        self,
        config_path: str = "/Users/hakeemshindy/Desktop/CryptoSentiment/CryptoSentiment_repo/config.yaml",
        *,
        include_labels: bool = True,
        label_reducer: Optional[Callable[[pd.Series], float | int | str]] = None,
    ) -> None:
        self.include_labels = include_labels
        self.label_reducer = label_reducer or (lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])

        # ------------------------------------------------------------------
        # Read configuration
        # ------------------------------------------------------------------
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg.get("data", {})
        self.prebit_dataset_path: Optional[str] = data_cfg.get("prebit_dataset_path")
        self.prebit_dataset_dir: Optional[str] = data_cfg.get("prebit_dataset_dir")

        if not (self.prebit_dataset_path or self.prebit_dataset_dir):
            raise ValueError(
                "Specify either 'prebit_dataset_path' (single‑CSV) or 'prebit_dataset_dir' (directory) in config.yaml → data"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, *, aggregate: bool = False) -> pd.DataFrame:
        """Load tweets (+ price) with optional per‑day aggregation.

        Parameters
        ----------
        aggregate : bool, default ``False``
            If *True*, collapse multiple tweets on the same date into a single
            row.  Numerical columns are *mean*‑aggregated, string columns are
            concatenated with ``" \n"`` separators, and label columns use
            :pyattr:`label_reducer`.
        """
        if self.prebit_dataset_path:
            df = self._load_single_prebit()
        else:
            df = self._load_prebit_dir()

        if aggregate:
            df = self._aggregate_per_day(df)

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_single_prebit(self) -> pd.DataFrame:
        """Load the (older) *single‑CSV* version of the PreBit dataset."""
        csv_path = Path(self.prebit_dataset_path)
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
