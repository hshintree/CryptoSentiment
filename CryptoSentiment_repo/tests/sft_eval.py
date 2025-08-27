#!/usr/bin/env python3
"""
Evaluate all checkpoint models in checkpoints/cryptobert-bitcoin-sft/

This script evaluates all checkpoint models found in the specified directory
and compares their performance on the validation dataset.

Usage:
    python evaluate_models.py [checkpoint_dir] [validation_file] [sample_n]

Example:
    python evaluate_models.py checkpoints/cryptobert-bitcoin-sft data/#2val.csv 25000
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torch
import yaml
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Add the parent directory to path to import our existing modules
sys.path.append(str(Path(__file__).parent))
from gpu_scripts.preprocessor import Preprocessor
from gpu_scripts.market_labeler_ewma import MarketLabelerTBL, MarketFeatureGenerator
from gpu_scripts.model import Model
from gpu_scripts.train_one import _coerce_dates

# ------------------------------------------------------------------ #
# 1)  FIT PREPROCESSOR ON THE *TRAINING* SET ONLY                    #
# ------------------------------------------------------------------ #
EA_CSV = Path("data/#1train.csv")
ea_train = (pd.read_csv(EA_CSV)
            .rename(columns={"date": "Tweet Date"}))
ea_train["Tweet Date"] = pd.to_datetime(ea_train["Tweet Date"], errors="coerce")

# --------------------------------------------------------------------- #
# -------------------------- CLI ARGUMENTS ---------------------------- #
# --------------------------------------------------------------------- #
if len(sys.argv) < 2:
    raise SystemExit("Usage: evaluate_models.py <checkpoint_dir> "
                     "[validation_file=data/#2val.csv] [sample_n=25000]")

CHECKPOINT_DIR = Path(sys.argv[1])
CSV_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/#2val.csv"
SAMPLE_N = int(sys.argv[3]) if len(sys.argv) > 3 else 25_000

if not CHECKPOINT_DIR.exists():
    raise SystemExit(f"❌  {CHECKPOINT_DIR} not found")

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
    torch.device("mps") if getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"🔹 Running inference on {DEVICE}")

# ------------------------------------------------------------------#
#  Preprocess validation data using the same pipeline as training
# ------------------------------------------------------------------#
print("\n🔹 Preprocessing validation data...")

# Apply date coercion using existing function
VAL = _coerce_dates(VAL)

# Use existing preprocessing pipeline to handle leaky datasets
logger.info("🧮 Applying existing preprocessing pipeline...")

# Initialize preprocessor and fit on training data only
pre = Preprocessor("config.yaml")
df_train_proc = pre.fit_transform(ea_train.copy())

# Apply fitted preprocessor to validation data
VAL_proc = pre.transform(VAL.copy())

# Apply TBL labeling
logger.info("🔬 Applying TBL labeling...")
labeler = MarketLabelerTBL("config.yaml", verbose=False)
df_train_lab = labeler.fit_and_label(df_train_proc)
VAL_lab = labeler.apply_labels(VAL_proc)

# Generate causal Previous Label
logger.info("🎯 Generating causal Previous Label...")
feat_gen = MarketFeatureGenerator()
VAL_lab["Previous Label"] = feat_gen.transform(VAL_lab)

# Calculate RSI and ROC for the prompts
print("🔹 Calculating RSI and ROC indicators...")
from ta.momentum import RSIIndicator, ROCIndicator

# Calculate RSI
rsi_indicator = RSIIndicator(close=VAL_lab['Close'], window=14)
VAL_lab['RSI'] = rsi_indicator.rsi()

# Calculate ROC
roc_indicator = ROCIndicator(close=VAL_lab['Close'], window=7)
VAL_lab['ROC'] = roc_indicator.roc()

# Bucket the indicators
def bucket_rsi(val: float) -> str:
    if pd.isna(val):
        return "neutral"
    if val < 30:
        return "oversold"
    if val > 70:
        return "overbought"
    return "neutral"

def bucket_roc(val: float, thr: float = 0.0, sigma: float = 2.0) -> str:
    if pd.isna(val):
        return "neutral"
    if val > thr + sigma:
        return "bullish"
    if val < thr - sigma:
        return "bearish"
    return "neutral"

VAL_lab['RSI_bucket'] = VAL_lab['RSI'].apply(bucket_rsi)
VAL_lab['ROC_bucket'] = VAL_lab['ROC'].apply(bucket_roc)

# Build prompts
def build_prompt(row: pd.Series) -> str:
    """Construct the pure‑text prompt (no date)."""
    return (
        f"Previous Label: {row['Previous Label']}, "
        f"RSI: {row['RSI_bucket']}, ROC: {row['ROC_bucket']}, "
        f"Tweet: {row['Tweet Content']}\nAssistant:"
    )

VAL_lab["text"] = VAL_lab.apply(build_prompt, axis=1)

# --------------------------------------------------------------------- #
# ------------------------- FIND CHECKPOINTS -------------------------- #
# --------------------------------------------------------------------- #
print(f"\n🔹 Finding checkpoints in {CHECKPOINT_DIR}...")

# Find all checkpoint directories
checkpoint_dirs = []
for item in CHECKPOINT_DIR.iterdir():
    if item.is_dir() and item.name.startswith("checkpoint-"):
        checkpoint_dirs.append(item)

# Also include the main directory if it has a model
if (CHECKPOINT_DIR / "model.safetensors").exists() or (CHECKPOINT_DIR / "pytorch_model.bin").exists():
    checkpoint_dirs.append(CHECKPOINT_DIR)

checkpoint_dirs.sort(key=lambda x: x.name)

if not checkpoint_dirs:
    raise SystemExit(f"❌  No checkpoint directories found in {CHECKPOINT_DIR}")

print(f"🔹 Found {len(checkpoint_dirs)} checkpoints:")
for ckpt in checkpoint_dirs:
    print(f"   - {ckpt.name}")

# --------------------------------------------------------------------- #
# ----------------------- EVALUATE MODELS ----------------------------- #
# --------------------------------------------------------------------- #
results = []

for checkpoint_dir in tqdm(checkpoint_dirs, desc="Evaluating checkpoints"):
    print(f"\n🔹 Evaluating {checkpoint_dir.name}...")
    
    try:
        # Load model
        model = Model("config.yaml")
        model.bert_model.from_pretrained(checkpoint_dir)
        model.tokenizer.from_pretrained(checkpoint_dir)
        model.device = DEVICE
        model.bert_model.to(DEVICE)
        
        # Tokenize validation data
        def tokenize_dataset(df: pd.DataFrame, tokenizer) -> list:
            """Tokenize the dataset for inference."""
            # Set a safe max length to avoid "int too big to convert" error
            safe_max_length = min(tokenizer.model_max_length or 512, 512)
            print(f"   Using max_length: {safe_max_length}")
            
            # Tokenize in batches to avoid memory issues
            batch_size = 16  # Reduced batch size for safety
            all_tokens = []
            
            for i in tqdm(range(0, len(df), batch_size), desc="Tokenizing", leave=False):
                batch_texts = df["text"].iloc[i:i+batch_size].tolist()
                try:
                    batch_tokens = tokenizer(
                        batch_texts,
                        truncation=True,
                        padding="max_length",
                        max_length=safe_max_length,
                        return_tensors="pt"
                    )
                    all_tokens.append(batch_tokens)
                except Exception as e:
                    print(f"   ⚠️  Error tokenizing batch {i//batch_size}: {e}")
                    # Try with shorter texts
                    shorter_texts = [text[:safe_max_length//2] if len(text) > safe_max_length//2 else text 
                                   for text in batch_texts]
                    batch_tokens = tokenizer(
                        shorter_texts,
                        truncation=True,
                        padding="max_length",
                        max_length=safe_max_length,
                        return_tensors="pt"
                    )
                    all_tokens.append(batch_tokens)
            
            return all_tokens
        
        # Tokenize validation data
        tokenized_data = tokenize_dataset(VAL_lab, model.tokenizer)
        
        # Run inference
        model.bert_model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_tokens in tqdm(tokenized_data, desc="Inference", leave=False):
                # Move to device
                batch_tokens = {k: v.to(DEVICE) for k, v in batch_tokens.items()}
                
                # Get predictions
                outputs = model.bert_model(**batch_tokens)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        # Map predictions back to labels
        LABEL_MAP_REVERSE = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
        pred_labels = [LABEL_MAP_REVERSE[pred] for pred in predictions]
        
        # Calculate metrics if we have true labels
        if "Label" in VAL_lab.columns:
            yt = VAL_lab["Label"].map({'Bearish': 0, 'Neutral': 1, 'Bullish': 2}).values
            yp = np.array([LABEL_MAP_REVERSE[pred] for pred in predictions])
            yp = np.array([{'Bearish': 0, 'Neutral': 1, 'Bullish': 2}[pred] for pred in yp])
            
            acc = accuracy_score(yt, yp)
            prec, rec, f1, _ = precision_recall_fscore_support(
                yt, yp, average="macro", zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
            
            # Feature correlations
            feats = ["RSI", "ROC", "Previous Label"]
            corr_data = {}
            if all(c in VAL_lab.columns for c in feats):
                enc = VAL_lab.replace({"Bearish": 0, "Neutral": 1, "Bullish": 2})
                pred_df = pd.DataFrame({"Pred_Label": pred_labels})
                enc = pd.concat([enc[feats], pred_df], axis=1)
                enc["Pred_Label"] = enc["Pred_Label"].map({"Bearish": 0, "Neutral": 1, "Bullish": 2})
                corr = enc[feats + ["Pred_Label"]].corr().loc[feats, "Pred_Label"]
                corr_data = corr.abs().sort_values(ascending=False).to_dict()
            
            result = {
                "checkpoint": checkpoint_dir.name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm.tolist(),
                "feature_correlations": corr_data,
                "prediction_distribution": pd.Series(pred_labels).value_counts().to_dict(),
                "true_distribution": VAL_lab["Label"].value_counts().to_dict() if "Label" in VAL_lab.columns else {}
            }
        else:
            result = {
                "checkpoint": checkpoint_dir.name,
                "prediction_distribution": pd.Series(pred_labels).value_counts().to_dict()
            }
        
        results.append(result)
        
        # Save individual predictions
        VAL_with_preds = VAL_lab.copy()
        VAL_with_preds["Pred_Label"] = pred_labels
        output_file = f"predictions_{checkpoint_dir.name}.csv"
        VAL_with_preds.to_csv(output_file, index=False)
        print(f"   ✅ Predictions saved to {output_file}")
        
        # Print metrics
        if "Label" in VAL_lab.columns:
            print(f"   ⚡ Accuracy : {acc:.3f}")
            print(f"   ⚡ Precision: {prec:.3f}")
            print(f"   ⚡ Recall   : {rec:.3f}")
            print(f"   ⚡ F1-score : {f1:.3f}")
            
            print("   ⚡ Confusion-matrix rows=truth, cols=pred")
            print(f"   {cm}")
            
            if corr_data:
                print("   ⚡ ρ(feature , Pred_Label)")
                for k, v in corr_data.items():
                    print(f"      {k:<14}: {v:+.3f}")
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"   ❌ Error evaluating {checkpoint_dir.name}: {e}")
        results.append({
            "checkpoint": checkpoint_dir.name,
            "error": str(e)
        })

# --------------------------------------------------------------------- #
# ------------------------- SAVE RESULTS ------------------------------ #
# --------------------------------------------------------------------- #
# Save all results to JSON
with open("checkpoint_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to checkpoint_evaluation_results.json")

# Print summary
print(f"\n📊 SUMMARY")
print(f"🗳️  checkpoints evaluated: {len([r for r in results if 'error' not in r])}")
print(f"📄  validation samples: {len(VAL):,}")

if any("f1_score" in r for r in results):
    print(f"\n🏆 Best F1 Score:")
    best_result = max([r for r in results if "f1_score" in r], key=lambda x: x["f1_score"])
    print(f"   {best_result['checkpoint']}: {best_result['f1_score']:.3f}")

    print(f"\n📈 All Results (sorted by F1):")
    sorted_results = sorted([r for r in results if "f1_score" in r], 
                           key=lambda x: x["f1_score"], reverse=True)
    for result in sorted_results:
        print(f"   {result['checkpoint']:<20}: F1={result['f1_score']:.3f}, "
              f"Acc={result['accuracy']:.3f}, Prec={result['precision']:.3f}, "
              f"Rec={result['recall']:.3f}")