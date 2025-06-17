from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd
import yaml


def _find_close_col(df: pd.DataFrame) -> str:
    """Return the name of the column that stores the daily *close* price."""
    for candidate in ("Close_x", "close", "Close"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not identify close price column (expected one of Close_x/close/Close)"
    )


def _discover_label_cols(cols: Iterable[str]) -> List[str]:
    """Return all column names that start with the Kaggle-style label prefix."""
    return [c for c in cols if c.startswith("label_")]


class DatasetLoader:
    """Load PreBit multimodal dataset (tweets + price + labels)."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        *,
        include_labels: bool = True,
        label_reducer: Optional[Callable[[pd.Series], float | int | str]] = None,
    ) -> None:
        self.include_labels = include_labels
        self.label_reducer = label_reducer or (
            lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]
        )

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg.get("data", {})
        self.prebit_dataset_path: Optional[str] = data_cfg.get("prebit_dataset_path")
        self.prebit_dataset_dir: Optional[str] = data_cfg.get("prebit_dataset_dir")

        if not (self.prebit_dataset_path or self.prebit_dataset_dir):
            raise ValueError(
                "Specify either 'prebit_dataset_path' (single-CSV) or 'prebit_dataset_dir' (directory) in config.yaml → data"
            )

    def load_dataset(self, *, aggregate: bool = False) -> pd.DataFrame:
        """Load tweets (+ price) with optional per-day aggregation."""
        if self.prebit_dataset_path:
            df = self._load_single_prebit()
        else:
            df = self._load_prebit_dir()

        if aggregate:
            df = self._aggregate_per_day(df)
        return df

    def _load_single_prebit(self) -> pd.DataFrame:
        csv_path = Path(self.prebit_dataset_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path, parse_dates=["date"])
        required = {"date", "tweet", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"Expected at least columns {required}, got {set(df.columns)}")

        if not self.include_labels:
            df = df.drop(columns=_discover_label_cols(df.columns), errors="ignore")

        return df.rename(
            columns={"date": "Tweet Date", "tweet": "Tweet Content", "close": "Close"}
        )

    def _load_prebit_dir(self) -> pd.DataFrame:
        dir_path = Path(self.prebit_dataset_dir)
        if not dir_path.exists():
            raise FileNotFoundError(dir_path)

        tweet_csvs = sorted(dir_path.glob("combined_tweets_*_labeled.csv"))
        if not tweet_csvs:
            raise FileNotFoundError(
                "No tweet CSVs matching 'combined_tweets_*_labeled.csv' found in "
                + str(dir_path)
            )

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

        price_path = dir_path / "price_label.csv"
        if not price_path.exists():
            return tweets

        price_df = pd.read_csv(price_path, parse_dates=["date"])
        close_col = _find_close_col(price_df)

        price_keep = ["date", close_col]
        if self.include_labels:
            price_keep.extend(_discover_label_cols(price_df.columns))

        price_df = price_df[price_keep]
        price_df = price_df.rename(columns={"date": "Tweet Date", close_col: "Close"})

        merged = tweets.merge(price_df, on="Tweet Date", how="left", suffixes=("", "_price"))
        return merged

    def _aggregate_per_day(self, df: pd.DataFrame) -> pd.DataFrame:
        label_cols = _discover_label_cols(df.columns) if self.include_labels else []
        numeric_cols = df.select_dtypes("number").columns.difference(label_cols)

        def _agg_fn(series: pd.Series):
            if series.name in label_cols:
                return self.label_reducer(series)
            if series.name in numeric_cols:
                return series.mean()
            if series.name == "Tweet Content":
                return " \n".join(series.astype(str))
            return series.iloc[0]

        grouped = df.groupby("Tweet Date", as_index=False).agg(_agg_fn)
        return grouped
