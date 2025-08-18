#!/usr/bin/env python3
"""
Leak‑free dataset generator (v4)
================================
* Keeps the **original EvalLoader logic** you liked, but:
  • uses the *causal* `MarketFeatureGenerator` for `Previous Label`.
  • eliminates CV‐fold output (Ea/Eb only).
  • writes **data/#2train.csv**  (Ea‑2020) and **data/#3val.csv**  (Eb 2015‑18 & 2022‑23).
* Robust date parsing so 2022‑2023 aren’t dropped.
* No overwriting of your existing #1train / #2val files.

Run:
    python generate_eval_datasets_ca.py --raw data/combined_dataset_raw.csv
"""
import argparse, warnings
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np
from tqdm import tqdm

from market_labeler_ewma import MarketLabelerTBL, MarketFeatureGenerator

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

YEAR_EA = [2020]
YEAR_EB = [2015, 2016, 2017, 2018, 2022, 2023]


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Robust date parse – strip tz & weird separators
    df["date"] = (df["date"].astype(str)
                    .str.replace(r"[TZ].*$", "", regex=True)
                    .str.replace(r"\\.", "-", regex=True))
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def label_and_prev(df: pd.DataFrame, cfg="config.yaml", verbose=False):
    pre_lab = MarketLabelerTBL(cfg, verbose=verbose).fit_and_label(df)
    prev = MarketFeatureGenerator().fit().transform(
        pre_lab.rename(columns={"date": "Tweet Date"})
    )
    pre_lab["Previous Label"] = prev
    # Strip barriers / vol to avoid leakage downstream
    drop_cols = [c for c in pre_lab.columns if c.lower().startswith(("upper","lower","vertical","volatility","barrier"))]
    return pre_lab.drop(columns=drop_cols)


def build_ea_eb(df_lab: pd.DataFrame):
    ea = df_lab[df_lab["date"].dt.year.isin(YEAR_EA)].copy()
    eb = df_lab[df_lab["date"].dt.year.isin(YEAR_EB)].copy()

    # Balance Ea (~20k each class if possible)
    target = 60000
    grp_sz = target // 3
    ea_bal = pd.concat([
        g.sample(min(len(g), grp_sz), random_state=42)
        for _, g in ea.groupby("Label")
    ])
    ea_bal = ea_bal.sort_values("date").reset_index(drop=True)

    # Eb: cap at 40k, keep ≥10 % events if available
    evt = eb[eb["Is_Event"] == 1]
    non = eb[eb["Is_Event"] == 0]
    n_evt = min(len(evt), 20000)
    n_non = min(len(non), 40000 - n_evt)
    eb_samp = pd.concat([
        evt.sample(n_evt, random_state=42) if n_evt else evt,
        non.sample(n_non, random_state=42) if n_non else non,
    ])
    eb_samp = eb_samp.sort_values("date").reset_index(drop=True)
    return ea_bal, eb_samp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="combined_dataset_raw.csv")
    p.add_argument("--cfg", default="config.yaml")
    args = p.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        raise SystemExit(f"❌ {raw} not found")

    print("🚀 Loading & cleaning raw data…")
    df = load_clean(raw)
    print(f"Rows after clean: {len(df):,}; years: {sorted(df['date'].dt.year.unique())}")

    print("🪄 Labeling with EWMA‑TBL + causal prev‑label…")
    df_lab = label_and_prev(df, args.cfg)

    print("📦 Building Ea / Eb splits…")
    ea, eb = build_ea_eb(df_lab)
    print(f"Ea 2020 balanced: {len(ea):,} rows  | dist: {ea['Label'].value_counts().to_dict()}")
    print(f"Eb 2015‑18/22‑23: {len(eb):,} rows  | dist: {eb['Label'].value_counts().to_dict()}")

    out_dir = Path("data"); out_dir.mkdir(exist_ok=True)
    ea.to_csv(out_dir/"#2train.csv", index=False)
    eb.to_csv(out_dir/"#3val.csv",   index=False)
    print("💾 Saved: #2train.csv  #3val.csv")

    # Quick rho sanity
    for name, d in [("train", ea), ("val", eb)]:
        rho = (d.replace({'Bearish':0,'Neutral':1,'Bullish':2})[["Label","Previous Label"]]
                 .corr().iloc[0,1])
        print(f"ρ(Label,Prev) {name}: {rho:.3f}")

    print("✅ Finished", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
