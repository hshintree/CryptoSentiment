#!/usr/bin/env python3
"""
Inference helper that ensembles the per-fold checkpoints produced by
tests/full_cv_run.py.

Typical use
-----------
    # VAL (≈435 k tweets) with the four folds saved at 07:47:12 on 21 Jun 2025
    python checkpointed_inference.py 20250621_074712

    # Same, but sub-sample to 50 k tweets
    python checkpointed_inference.py 20250621_074712 50000

    # Run on Eb instead of VAL
    python checkpointed_inference.py 20250621_074712 0 data/#2val.csv
"""

import sys, json, numpy as np, pandas as pd
from pathlib import Path
from glob     import glob
from tqdm     import tqdm
import torch
import yaml
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support,
                             confusion_matrix)

# ── Crypto-Sentiment code -----------------
from gpu_scripts.model   import Model
from gpu_scripts.trainer import Trainer

# --------------------------------------------------------------------- #
# -------------------------- CLI ARGUMENTS ---------------------------- #
# --------------------------------------------------------------------- #
if len(sys.argv) < 2:
    raise SystemExit("Usage:  checkpointed_inference.py <timestamp>  "
                     "[sample_n=25000]  [csv_path]  [--no-save]  [--timestamped]")

TS         = sys.argv[1]                 # e.g. 20250621_074712
SAMPLE_N   = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 25_000
CSV_PATH   = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "data/#2val.csv"
NO_SAVE    = "--no-save" in sys.argv
TIMESTAMPED= "--timestamped" in sys.argv

# ------------------------------------------------------------------ #
# 1)  FIT PREPROCESSOR ON THE *TRAINING* SET ONLY                    #
# ------------------------------------------------------------------ #
EA_CSV   = Path("data/#1train.csv")
ea_train = (pd.read_csv(EA_CSV)
              .rename(columns={"date": "Tweet Date"}))
ea_train["Tweet Date"] = pd.to_datetime(ea_train["Tweet Date"], errors="coerce")

# --------------------------------------------------------------------- #
# ------------------------------ DATA --------------------------------- #
# --------------------------------------------------------------------- #
val_files = sorted(glob(CSV_PATH))
if not val_files:
    raise SystemExit(f"❌  dataset '{CSV_PATH}' not found")

VAL = (pd.read_csv(val_files[-1])
         .rename(columns={"date": "Tweet Date"}))
VAL["Tweet Date"] = pd.to_datetime(VAL["Tweet Date"],
                                   format="mixed", errors="coerce")

if SAMPLE_N and len(VAL) > SAMPLE_N:
    tgt = int(np.ceil(SAMPLE_N / 12))
    frames = [m.sample(n=min(tgt, len(m)), random_state=42)
              for _, m in VAL.groupby(VAL["Tweet Date"].dt.to_period("M"))]
    VAL = (pd.concat(frames)
             .sort_values("Tweet Date")
             .reset_index(drop=True))
    print(f"🔹 Stratified sample: {len(VAL):,} tweets ({tgt}/month max)")
else:
    print(f"🔹 Using full set: {len(VAL):,} tweets")

# --------------------------------------------------------------------- #
# ---------------------------- DEVICE --------------------------------- #
# --------------------------------------------------------------------- #
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if getattr(torch.backends, "mps", None)
                           and torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"🔹 Running inference on {DEVICE}")

# ------------------------------------------------------------------#
#  Trainer object that knows how to preprocess & ensemble
# ------------------------------------------------------------------#
cfg = yaml.safe_load(open("config.yaml"))
base_model = Model(cfg["model"])
trainer = Trainer(base_model,  # dummy model just for preprocessing
                  ea_train,
                  "config.yaml",
                  quiet=True)
trainer.device = DEVICE
base_model.device = DEVICE
base_model.bert_model.to(DEVICE)
trainer.fold_states = []          # we'll fill it next

print("\n🔹 Loading fold checkpoints …")

# ------------------------------------------------------------------
# Which folds should participate in the ensemble?
#   • Give an explicit set → we will load only those
#   • Leave the set empty  → load every folder that exists
# ------------------------------------------------------------------
#USE_FOLDS = {"fold2"}               # ← EXAMPLE 2  (strongest only)
USE_FOLDS = set()                     # finds all available fold directories

# We do **not** hand-craft prompts or soft-max any more – the
# Trainer object will do leakage-safe preprocessing + averaging.

# ------------------------------------------------------------------
# Resolve folders – accept either:
#   • fold2/                           (plain)
#   • fold2_epoch3/                    (epoch-suffixed)
# ------------------------------------------------------------------
from glob import glob as _g
CKPT_ROOT  = Path("models") / f"ea_"
if not CKPT_ROOT.exists():
    raise SystemExit(f"❌  {CKPT_ROOT} not found")

def _resolve(fold_name: str) -> Path | None:
    """Return the directory that actually exists for this fold."""
    plain = CKPT_ROOT / fold_name
    if plain.exists():
        return plain
    # fallback: first match with the prefix, e.g. fold2_epoch*
    matches = sorted(CKPT_ROOT.glob(f"{fold_name}_epoch*"))
    return matches[-1] if matches else None

if USE_FOLDS:
    candidate_dirs = [p for f in USE_FOLDS
                        if (p := _resolve(f)) is not None]
else:
    # grab *all* fold-prefixes, regardless of suffix
    candidate_dirs = sorted({p.parent          # remove /epochX
                             for p in CKPT_ROOT.glob("fold[0-9]*_epoch*")}
                            | set(CKPT_ROOT.glob("fold[0-9]*")))

for fold_dir in candidate_dirs:
    chk_bin  = fold_dir / "pytorch_model.bin"
    chk_safe = fold_dir / "model.safetensors"
    if not (chk_bin.exists() or chk_safe.exists()):
        print(f"⚠️  {fold_dir.name}: no model file → skip")
        continue

    m = Model(cfg["model"])
    m.bert_model.from_pretrained(fold_dir)
    m.tokenizer.from_pretrained(fold_dir)
    m.device = DEVICE
    m.bert_model.to("cpu")           # keep off-GPU until used

    # try to fetch best-fold F1 for weighting; fall back to 1.0
    mjson = fold_dir / "metrics.json"
    val_f1 = 1.0
    if mjson.exists():
        try:
            with open(mjson) as fh:
                val_f1 = json.load(fh).get("val_f1", 1.0)
        except Exception:
            pass

    trainer.fold_states.append({"model": m, "val_f1": val_f1})
    print(f"   ✓ added {fold_dir.name}  (val_F1={val_f1:.3f})")

if not trainer.fold_states:
    raise SystemExit("❌  no usable folds found!")

print("\n🔹 Running softmax-averaged ensemble …")
VAL = trainer.ensemble_predict(VAL, weighted=True)

suffix = "_" + TS if TIMESTAMPED else ""

if not NO_SAVE:
    VAL.to_csv(f"signals_val_2021{suffix}.csv", index=False)
    print(f"✅  signals_val_2021{suffix}.csv written")

# ---- safe daily aggregation ---------------------------------------- #
def _majority_or_neutral(series):
    non_na = series.dropna()
    return (non_na.value_counts().idxmax()
            if len(non_na) else "Neutral")

daily = (VAL.groupby(VAL["Tweet Date"].dt.normalize()).Pred_Label
           .agg(_majority_or_neutral)
           .rename("Daily_Signal"))

if not NO_SAVE:
    daily.to_csv(f"val_daily_signals{suffix}.csv")
    print(f"✅  val_daily_signals{suffix}.csv written")

# metrics
if "Label" in VAL.columns:
    yt = VAL["Label"].map({'Bearish':0,'Neutral':1,'Bullish':2}).values
    yp = VAL["Pred_Label"].map({'Bearish':0,'Neutral':1,'Bullish':2}).values

    acc  = accuracy_score(yt, yp)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0)

    print(f"⚡  Accuracy : {acc:.3f}")
    print(f"⚡  Precision: {prec:.3f}")
    print(f"⚡  Recall   : {rec:.3f}")
    print(f"⚡  F1-score : {f1:.3f}")

    # --- extra diagnostics -------------------------------------------------
    cm = confusion_matrix(yt, yp, labels=[0,1,2])
    print("⚡  Confusion-matrix rows=truth, cols=pred")
    print(cm)

    # feature⇆label Pearson (RSI / ROC / Prev-Label) on Eb predictions
    feats = ["RSI","ROC","Previous Label"]
    if all(c in VAL.columns for c in feats):
        enc = VAL.replace({"Bearish":0,"Neutral":1,"Bullish":2})
        corr = enc[feats+["Pred_Label"]].corr().loc[feats,"Pred_Label"]
        print("⚡  ρ(feature , Pred_Label)   (Eb set)")
        for k,v in corr.abs().sort_values(ascending=False).items():
            print(f"   {k:<14}: {v:+.3f}")

# --------------------------------------------------------------------- #
# -------------------------- SMALL SUMMARY --------------------------- #
# --------------------------------------------------------------------- #
print(f"🗳️  folds ensembled : {len(trainer.fold_states)}")
print(f"📄  daily rows      : {len(daily):,}")
print("Pred distribution :", VAL['Pred_Label'].value_counts().to_dict())
print(VAL['Pred_Label'].value_counts(normalize=True))

print("True distribution :", VAL['Label'].value_counts().to_dict())
print(VAL['Label'].value_counts(normalize=True))