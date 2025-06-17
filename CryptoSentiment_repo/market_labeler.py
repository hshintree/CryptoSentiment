"""market_labeler.py

Triple‑Barrier *adaptive* labeler.

Implements the **6‑monthly grid‑search** described in
"Revisiting Financial Sentiment Analysis: A Language Model Approach" – the
barrier factors (Fu, Fl) and the vertical barrier horizon (Vt) are
re‑optimised every ~126 trading days (≈ 6 calendar months) to maximise the
in‑sample Sharpe ratio of a naïve long/short strategy that enters on every
signal.

The grid size is intentionally small so optimisation remains fast – feel free
to widen the ranges in the YAML config.
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
    fu_grid: list[float]  # e.g. [0.02, 0.04, 0.06]
    fl_grid: list[float]  # e.g. [0.02, 0.04, 0.06]
    vt_grid: list[int]    # e.g. [3, 5, 10]
    rebalance_days: int   # optimise every N obs (≈ 126 for 6 months)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "_TBLConfig":
        cfg = yaml.safe_load(open(path))
        mkt = cfg.get("market_labeling", {})
        return cls(
            fu_grid=mkt.get("fu_grid", [0.04, 0.06, 0.08]),
            fl_grid=mkt.get("fl_grid", [0.04, 0.06, 0.08]),
            vt_grid=mkt.get("vt_grid", [5, 10]),
            rebalance_days=mkt.get("rebalance_days", 126),
        )

# ---------------------------------------------------------------------------
# Helper – Sharpe of naïve strategy -----------------------------------------
# ---------------------------------------------------------------------------

def _sharpe(returns: pd.Series) -> float:
    if returns.std(ddof=0) == 0:
        return -np.inf
    return returns.mean() / returns.std(ddof=0) * np.sqrt(252)

# ---------------------------------------------------------------------------
# Main labeler ---------------------------------------------------------------
# ---------------------------------------------------------------------------

class MarketLabeler:
    def __init__(self, cfg_path: str = "config.yaml") -> None:
        self.cfg = _TBLConfig.from_yaml(cfg_path)

    # ------------------------------------------------------------------
    def label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Upper/Lower/Vertical Barrier + categorical *Label* per row.

        Parameters
        ----------
        data : DataFrame with at least columns ``Tweet Date`` & ``Close``.
        """
        df = data.copy().sort_values("Tweet Date").reset_index(drop=True)

        prices = df["Close"].values
        dates  = pd.to_datetime(df["Tweet Date"]).values

        # pre‑allocate
        upper, lower, label = np.empty(len(df)), np.empty(len(df)), np.empty(len(df), dtype=object)

        # --- rolling optimisation -------------------------------------
        n = len(df)
        step = self.cfg.rebalance_days
        for start in range(0, n, step):
            end = min(start + step, n)
            win_prices = prices[start:end]

            # grid‑search on the *previous* window, except for the very first one
            opt_slice = slice(max(0, start - step), start)
            fu, fl, vt = self._optimize_params(prices[opt_slice])

            # apply TBL with optimal params to current slice
            u, l, lab = self._apply_tbl(win_prices, fu, fl, vt)
            upper[start:end], lower[start:end], label[start:end] = u, l, lab

        df["Upper Barrier"] = upper
        df["Lower Barrier"] = lower
        df["Vertical Barrier"] = vt  # constant within slice (last vt)
        df["Label"] = label
        return df

    # ------------------------------------------------------------------
    def _optimize_params(self, prices: np.ndarray) -> tuple[float, float, int]:
        """Grid‑search Fu, Fl, Vt on *in‑sample* slice to max Sharpe."""
        best = (-np.inf, self.cfg.fu_grid[0], self.cfg.fl_grid[0], self.cfg.vt_grid[0])
        for fu, fl, vt in itertools.product(self.cfg.fu_grid, self.cfg.fl_grid, self.cfg.vt_grid):
            _, _, lab = self._apply_tbl(prices, fu, fl, vt)
            rets = self._strategy_returns(prices, lab)
            s = _sharpe(pd.Series(rets))
            if s > best[0]:
                best = (s, fu, fl, vt)
        return best[1:]

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_tbl(prices: np.ndarray, fu: float, fl: float, vt: int):
        """Vectorised triple‑barrier."""
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
                # whichever comes first in time wins
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

    # ------------------------------------------------------------------
    @staticmethod
    def _strategy_returns(prices: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Generate simple daily returns for long/short strategy used in Sharpe optimisation."""
        pos = np.where(labels == "Bullish", +1, np.where(labels == "Bearish", -1, 0))
        pct = np.diff(prices) / prices[:-1]
        # align returns with starting position (length n‑1)
        return pos[:-1] * pct
