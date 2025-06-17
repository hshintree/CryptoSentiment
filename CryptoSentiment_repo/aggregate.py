#!/usr/bin/env python3
"""
Compile PreBit yearly tweet CSVs + price_label.csv into one file.

Usage:
    python aggregate.py config.yaml out/compiled_prebit.csv

What it does: Reads the directory specified in config.yaml → data → prebit_dataset_dir.
Uses the DatasetLoader merge logic (tweets + price) so the compiled file already has the Close column and any price-level labels.
Writes a single CSV wherever you point it.
After that, set yaml data:
  prebit_dataset_path: "/path/to/out/compiled_prebit.csv"
  prebit_dataset_dir: null            # or just delete the key
and everything else in your pipeline keeps working.
"""
import sys
from pathlib import Path
import pandas as pd
from CryptoSentiment_repo.dataset_old_loader import DatasetLoader   # the canvas file you already have

def main(cfg_path: str, out_csv: str):
    dl = DatasetLoader(cfg_path)
    df = dl.load_dataset(aggregate=False)   # keep every tweet row
    out_csv = Path(out_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ wrote {len(df):,} rows → {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: aggregate.py CONFIG.YAML OUTPUT_CSV")
    main(sys.argv[1], sys.argv[2])
