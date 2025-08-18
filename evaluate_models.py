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
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support)
from preprocessor import Preprocessor
from market_labeler_ewma import MarketLabelerTBL, MarketFeatureGenerator
from model import Model                       # only for preprocess_input

# ------------------------------------------------------------------ #
# 1)  FIT PREPROCESSOR ON THE *TRAINING* SET ONLY                    #
# ------------------------------------------------------------------ #
EA_CSV     = Path("data/#1train.csv")
ea_train   = pd.read_csv(EA_CSV)
ea_train["date"] = pd.to_datetime(ea_train["date"], errors="coerce")

pre        = Preprocessor("config.yaml")
ea_train   = pre.fit(ea_train)      # fit scaler, no leak

# prepare labeler+feature generator for Previous-Label
tbl        = MarketLabelerTBL("config.yaml")
ea_train   = tbl.fit_and_label(ea_train)
feat_gen   = MarketFeatureGenerator("config.yaml")
feat_gen.fit(ea_train)

# dummy model instance – only used for .preprocess_input()
dummy_model = Model(yaml.safe_load(open("config.yaml"))["model"])

# --------------------------------------------------------------------- #
# -------------------------- CLI ARGUMENTS ---------------------------- #
# --------------------------------------------------------------------- #
if len(sys.argv) < 2:
    raise SystemExit("Usage:  checkpointed_inference.py <timestamp>  "
                     "[sample_n=25000]  [csv_path]")

TS         = sys.argv[1]                 # e.g. 20250621_074712
SAMPLE_N   = int(sys.argv[2]) if len(sys.argv) > 2 else 25_000
CSV_PATH   = sys.argv[3] if len(sys.argv) > 3 else "data/#2val.csv"

CKPT_ROOT  = Path("models") / f"ea_"
if not CKPT_ROOT.exists():
    raise SystemExit(f"❌  {CKPT_ROOT} not found")

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

# ------------------------------------------------------------------
# Which folds should participate in the ensemble?
#   • Give an explicit set → we will load only those
#   • Leave the set empty  → load every folder that exists
# ------------------------------------------------------------------
#USE_FOLDS = {"fold2", "fold3"}        # ← EXAMPLE 1  (drop 1 & 4)
#USE_FOLDS = {"fold2"}               # ← EXAMPLE 2  (strongest only)
USE_FOLDS = set()                     # finds all available fold directories

def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

probs = np.zeros((len(VAL), 3), dtype=float)
n_live_folds = 0

# ------------------------------------------------------------------
# Resolve folders – accept either:
#   • fold2/                           (plain)
#   • fold2_epoch3/                    (epoch-suffixed)
# ------------------------------------------------------------------
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
    bin_ok   = (fold_dir / "pytorch_model.bin").exists()
    safe_ok  = (fold_dir / "model.safetensors").exists()
    if not (bin_ok or safe_ok):
        print(f"⚠️  {fold_dir.name} – no model file, skipping")
        continue

    print(f"🔗 Loading {fold_dir.name} …")
    tok = AutoTokenizer.from_pretrained(fold_dir)
    mdl = (AutoModelForSequenceClassification
           .from_pretrained(fold_dir)
           .to(DEVICE)
           .eval())
    n_live_folds += 1
    bs = 256 if DEVICE.type in {"cuda", "mps"} else 128

    with torch.no_grad():
        for i in tqdm(range(0, len(VAL), bs), desc=fold_dir.name, ncols=80):
            chunk = VAL.iloc[i:i+bs]
            # ----------------------------------------------------------------
            # 2)  REBUILD THE EXACT PROMPT FOR **EACH TWEET** IN THE CHUNK
            # ----------------------------------------------------------------
            chunk_proc = pre.transform(chunk.copy())
            chunk_proc = tbl.apply_labels(chunk_proc)          # adds Label
            chunk_proc["Previous Label"] = feat_gen.transform(chunk_proc)

            prompts = [
                dummy_model.preprocess_input(
                    tweet_content = row["Tweet Content"],
                    rsi           = row["RSI"],
                    roc           = row["ROC"],
                    previous_label= row["Previous Label"],
                    as_prompt_only=True          # <- return plain string
                )
                for _, row in chunk_proc.iterrows()
            ]

            enc = tok(prompts,
                      padding=True, truncation=True, max_length=128,
                      return_tensors="pt")
            out   = mdl(**{k: v.to(DEVICE) for k, v in enc.items()})
            probs[i:i+bs] += _softmax(out.logits.cpu().numpy())

if n_live_folds == 0:
    raise SystemExit("❌  no usable folds found!")

# --------------------------------------------------------------------- #
# --------------------------- POST-PROCESS ---------------------------- #
# --------------------------------------------------------------------- #
probs /= n_live_folds          # normalize by number of participating folds
pred_ids = probs.argmax(-1)
labels   = np.array(["Bearish", "Neutral", "Bullish"])
VAL["Pred_Label"] = labels[pred_ids]
VAL["Pred_Conf"]  = probs.max(-1) / np.maximum(probs.sum(-1), 1e-9)

VAL.to_csv("signals_val_2021.csv", index=False)
print("✅  signals_val_2021.csv written")

# ---- safe daily aggregation ---------------------------------------- #
def _majority_or_neutral(series):
    non_na = series.dropna()
    return (non_na.value_counts().idxmax()
            if len(non_na) else "Neutral")

daily = (VAL.groupby(VAL["Tweet Date"].dt.normalize()).Pred_Label
           .agg(_majority_or_neutral)
           .rename("Daily_Signal"))

daily.to_csv("val_daily_signals.csv")
print("✅  val_daily_signals.csv written")

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
    from sklearn.metrics import confusion_matrix
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
print(f"🗳️  folds ensembled : {n_live_folds}")
print(f"📄  daily rows      : {len(daily):,}")
print("Pred distribution :", VAL['Pred_Label'].value_counts().to_dict())
print(VAL['Pred_Label'].value_counts(normalize=True))

print("True distribution :", VAL['Label'].value_counts().to_dict())
print(VAL['Label'].value_counts(normalize=True))