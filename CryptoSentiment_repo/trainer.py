import os
# enable MPS→CPU fallback for any unsupported kernels
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# trainer.py  — hot-fix: ensure numeric hyper-parameters are cast to float/int
import yaml
import json
import csv
import torch
# MPS can be fragile with float64—force defaults to float32
torch.set_default_dtype(torch.float32)
from torch import nn
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
from tqdm import tqdm
from pathlib import Path

from transformers import get_linear_schedule_with_warmup
try:
    from transformers.optimization import AdamW          # HF ≥4.40
except ImportError:
    from torch.optim import AdamW                        # fallback

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
# --- time-series split that is aware of "multiple rows per day" ----------
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict          # ← we already imported numpy once
from sklearn.metrics import confusion_matrix          # ➟ confusion-matrix

from model                 import Model
from market_labeler_ewma   import MarketLabelerTBL, MarketFeatureGenerator


class Trainer:
    """Handle training of the BERT-based model with grouped CV."""

    def __init__(self,
                 model: Model,
                 data: pd.DataFrame,
                 config_path: str = "config.yaml") -> None:

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path  # Store for fold-wise preprocessing
        self.model = model
        self.data  = data
        
        # ─── DEVICE SETUP ───────────────────────────────────────────
        # Prefer CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # if using MPS, force single-worker DataLoader to avoid pickle issues
        if self.device.type == "mps":
            torch.multiprocessing.set_sharing_strategy("file_system")
            
        # Override the model's detected device with trainer's device for consistency
        self.model.device = self.device

        # 2) move the whole HF model
        self.model.bert_model.to(self.device)

        # on MPS we need to convert the embedding weights
        if self.device.type == "mps":
            emb = self.model.bert_model.get_input_embeddings()
            # rewrap as a real, allocated Parameter on MPS
            emb.weight = nn.Parameter(emb.weight.to(self.device))
            # if you have prompt-tuning, do the same for your prompt embeddings
            if hasattr(self.model, "prompt_embeddings"):
                self.model.prompt_embeddings = nn.Parameter(
                    self.model.prompt_embeddings.to(self.device)
                )

        # 5) now recreate your optimizer so it sees real device tensors

        self.fold_states = []
        # ───────── NEW: container for confusion-matrices ──────────
        self.cm_history: list[dict] = []

        tcfg = self.config["training"]
        # ---- cast YAML scalars safely -----------------------------------
        self.learning_rate = float(tcfg.get("learning_rate", 1e-5))
        self.batch_size    = int  (tcfg.get("batch_size",    12))
        self.epochs        = int  (tcfg.get("epochs",        2))
        self.warmup_frac   = float(tcfg.get("warmup_steps",  0.1))
        # -----------------------------------------------------------------

        # Set reproducibility seeds
        seed = tcfg.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.optimizer = AdamW(self.model.bert_model.parameters(), lr=self.learning_rate)
    # ------------------------------------------------------------------  
    # Public API  
    # ------------------------------------------------------------------
    def train(self) -> None:
        """5-fold time-series CV + standard training loop."""
        data = self._prepare_data(self.data)

        # ---------- 5-fold **Blocked** Purged CV (contiguous chunks) -------
        gap_days   = 13
        unique_days = (
            self.data["Tweet Date"]
                .dt.normalize()
                .sort_values()
                .drop_duplicates()
                .reset_index(drop=True)
        )
        n_days   = len(unique_days)
        fold_len = n_days // 5

        # map each tweet → its 0-based day index
        day_lookup = pd.Series(
            np.arange(n_days, dtype=int),
            index=unique_days.values       # Timestamp → int
        )
        row_day_idx = day_lookup[self.data["Tweet Date"]
                                 .dt.normalize()].values

        def blocked_purged_splits():
            start = 0
            for _fold in range(5):
                stop = start + fold_len if _fold < 4 else n_days
                val_idx = np.arange(start, stop, dtype=int)

                # purge ± gap_days around every validation day-id
                purge_idx = {
                    d for v in val_idx
                    for d in range(max(0, v - gap_days),
                                   min(n_days, v + gap_days + 1))
                }
                gap_rows   = np.where(np.isin(row_day_idx, purge_idx))[0]
                train_rows = np.where(~np.isin(row_day_idx,
                                               list(val_idx) + list(purge_idx)))[0]
                val_rows   = np.where( np.isin(row_day_idx,  val_idx))[0]

                if len(val_rows) and len(train_rows):
                    yield train_rows, gap_rows, val_rows
                start = stop

        date_splitter = blocked_purged_splits()
        print(f"Using 5-fold **Blocked** Purged CV (gap_days={gap_days})")

        # ✅ TRAINING METRICS TRACKING
        fold_metrics = []
        
        for fold, (tr_idx, gap_idx, va_idx) in enumerate(date_splitter, 1):

            # pretty diagnostic of the actual temporal split
            train_dates = data.iloc[tr_idx]["Tweet Date"]
            val_dates   = data.iloc[va_idx]["Tweet Date"]
            print(f"Fold {fold}: "
                  f"Train span : {train_dates.min().date()} … {train_dates.max().date()}"
                  f"Val   span : {val_dates.min().date()} … {val_dates.max().date()}"
                  f"Gap (min)  : {gap_days} days\n")
            print(f"\n── Fold {fold}/5 ──")
            
            # ✅ VALIDATE NO TRAIN/VAL OVERLAP
            train_set = set(tr_idx)
            val_set = set(va_idx)
            assert not (train_set & val_set), f"❌ Fold {fold}: train/val overlap detected!"
            print(f"✅ Fold {fold}: No train/val overlap (train={len(train_set)}, val={len(val_set)})")
            
            # ✅ REINITIALIZE MODEL & OPTIMIZER FOR EACH FOLD
            print(f"🔄 Fold {fold}: Reinitializing model and optimizer...")
            
            # Create fresh model for this fold - force it to use trainer's device
            fold_model = Model(self.config["model"])
            # Override the model's detected device with trainer's device
            fold_model.device = self.device
            fold_model.bert_model.to(self.device)
            if self.device.type == "mps":
                emb = fold_model.bert_model.get_input_embeddings()
                emb.weight = nn.Parameter(emb.weight.to(self.device))
                if hasattr(fold_model, "prompt_embeddings"):
                    fold_model.prompt_embeddings = nn.Parameter(
                        fold_model.prompt_embeddings.to(self.device)
                    )
            
            # Create fresh optimizer for this fold - only trainable parameters
            if hasattr(fold_model, 'use_prompt_tuning') and fold_model.use_prompt_tuning:
                # Optimize prompt embeddings + last transformer layer + classifier for prompt-tuning
                trainable_params = list(fold_model.bert_model.classifier.parameters())
                trainable_params.extend(fold_model.bert_model.base_model.encoder.layer[-1].parameters())
                if hasattr(fold_model, 'prompt_embeddings'):
                    trainable_params.append(fold_model.prompt_embeddings)
                fold_optimizer = AdamW(trainable_params, lr=self.learning_rate)
                print(f"  🎯 Prompt-tuning: optimizing {sum(p.numel() for p in trainable_params)} parameters")
            else:
                # Standard fine-tuning
                fold_optimizer = AdamW(fold_model.bert_model.parameters(), lr=self.learning_rate)
            
            tr_df  = data.iloc[tr_idx].copy()          # will feed optimiser
            gap_df = data.iloc[gap_idx].copy()         # NEVER enters optimiser
            va_df  = data.iloc[va_idx].copy()
            
            # ✅ FOLD-WISE PREPROCESSING: Fit scaler on training data only
            print(f"  📊 Applying fold-wise preprocessing to prevent scaling leakage...")
            from preprocessor import Preprocessor
            fold_preprocessor = Preprocessor(self.config_path)
            
            # Fit preprocessor on training data and transform both splits
            tr_df  = fold_preprocessor.fit_transform(tr_df)
            if not gap_df.empty:                      # ← skip empty gap slices
                gap_df = fold_preprocessor.transform(gap_df)
            va_df  = fold_preprocessor.transform(va_df)
            
            # ✅ FOLD-WISE LABELING: Batch-label with TBL
            print(f"  🔬 Applying fold-wise TBL labeling…")
            fold_lbl = MarketLabelerTBL("config.yaml")
            tr_df  = fold_lbl.fit_and_label(tr_df)
            if not gap_df.empty:
                gap_df = fold_lbl.apply_labels(gap_df)
            va_df  = fold_lbl.apply_labels(va_df)

            # ✅ FOLD-WISE FEATURE: Generate causal 'Previous Label'
            print(f"  🎯 Generating causal previous-label feature…")
            # include *gap* rows when building the day→label map
            if not gap_df.empty:
                feat_gen = MarketFeatureGenerator().fit(
                    pd.concat([tr_df, gap_df], ignore_index=True)
                )
                gap_df["Previous Label"] = feat_gen.transform(gap_df)   # only for corr-checks
            else:
                feat_gen = MarketFeatureGenerator().fit(tr_df)
            tr_df["Previous Label"] = feat_gen.transform(tr_df)
            va_df["Previous Label"] = feat_gen.transform(va_df)

            # ── Correlations (raw & per-fold) ──────────────────────────────
            _code = {"Bearish": 0, "Neutral": 1, "Bullish": 2}

            # (A)  **fold-local**  — raw RSI/ROC (inverse-scaled)
            raw_val = va_df.copy()
            raw_val[["RSI", "ROC"]] = (
                fold_preprocessor.scaler.inverse_transform(
                    raw_val[["RSI", "ROC"]]
                )
            )
            encoded = raw_val.replace(_code)

            feature_cols = [
                "Previous Label", "RSI", "ROC", "Volume",
                "has_user_metadata", "has_tweet_metadata", "user_followers",
            ]
            feature_corrs = {}
            for col in feature_cols:
                if col not in encoded.columns:
                    continue            # skip missing columns gracefully
                try:
                    x = pd.to_numeric(encoded[col], errors="coerce")
                except Exception:
                    continue
                c = np.corrcoef(x.fillna(0), encoded["Label"])[0, 1]
                if not np.isnan(c):
                    feature_corrs[col] = c

            print("ρ(feature , Label) on *this fold*:")
            for k, v in sorted(feature_corrs.items(), key=lambda kv: -abs(kv[1])):
                print(f"   {k:<18}: {v:+.3f}")

            # keep the old single-feature print for consistency
            corr = feature_corrs.get("Previous Label", np.nan)
            print(f"ρ(Prev-Label , Label) = {corr:.3f}")
            # If you ever see ≳0.9 here you have found a leak!
            day_corr = (
                va_df
                .assign(day  = va_df["Tweet Date"].dt.normalize())
                .groupby("day")[["Previous Label", "Label"]]
                .first()                         # one row per day
                .replace({"Bearish": 0, "Neutral": 1, "Bullish": 2})
                .corr()
                .iloc[0, 1]
            )

            print(f"ρ_day-level = {day_corr:.3f}")   # paper ≈ 0.25–0.30

            # ── extra daily-level correlations for RSI/ROC ────────────────
            daily_val = (
                va_df.groupby(va_df["Tweet Date"].dt.normalize())
                     .first()                     # one row per day
                     .replace(_code)
            )
            for ind in ("RSI", "ROC"):
                if ind in daily_val.columns:
                    c = daily_val[[ind, "Label"]].corr().iloc[0, 1]
                    print(f"      ↳ {ind:3}  day-level corr : {c:+.3f}")

            # (B)  **optional EA-wide diagnostic** – now the safe way
            if fold == 1 and not hasattr(self, "_global_corr_done"):
                # --- 1. same preprocessing -------------------------------
                full_raw = fold_preprocessor.transform(data.copy())
                full_raw[["RSI", "ROC"]] = fold_preprocessor.scaler.inverse_transform(
                    full_raw[["RSI", "ROC"]]
                )

                # --- 2. label entire EA once (no peeking) ---------------
                ea_labeler = MarketLabelerTBL("config.yaml")
                full_raw   = ea_labeler.fit_and_label(full_raw)

                # --- 3. causal previous-label ---------------------------
                feat_gen_global = MarketFeatureGenerator()
                full_raw["Previous Label"] = feat_gen_global.transform(full_raw)

                # --- 4. day-level aggregates & correlations -------------
                daily_mean = (
                    full_raw
                    .groupby(full_raw["Tweet Date"].dt.normalize())
                    .agg({
                        "Previous Label": "first",   # regime is discrete
                        "RSI":            "mean",
                        "ROC":            "mean",
                        "Volume":         "mean",
                        "Label":          "first",
                    })
                    .replace(_code)
                )
                gcorr = (
                    daily_mean[["Previous Label", "RSI", "ROC", "Volume", "Label"]]
                    .corr()
                    .iloc[:-1, -1]    # everything versus ground-truth Label
                )

                print("\n🌐 ρ(feature , Label) on *entire EA* (day-level, mean-aggregated):")
                for k, v in gcorr.abs().sort_values(ascending=False).items():
                    print(f"   {k:<18}: {v:+.3f}")

                # extra print focused on RSI/ROC only
                print(f"      ↳ RSI  day-level corr : {gcorr['RSI']:+.3f}")
                print(f"      ↳ ROC  day-level corr : {gcorr['ROC']:+.3f}")

                self._global_corr_done = True
            
            # 🔍 DEBUG: Analyze data leakage potential
            print(f"\n  🔍 FOLD {fold} DEBUG ANALYSIS:")
            
            # Check date ranges
            tr_dates = tr_df["Tweet Date"].dt.date
            va_dates = va_df["Tweet Date"].dt.date
            print(f"     Train date range: {tr_dates.min()} to {tr_dates.max()}")
            print(f"     Val date range:   {va_dates.min()} to {va_dates.max()}")
            
            # Check for date overlap (should be NONE with proper temporal CV)
            date_overlap = set(tr_dates) & set(va_dates)
            if date_overlap:
                print(f"     ⚠️  DATE OVERLAP DETECTED: {len(date_overlap)} dates overlap!")
                print(f"         Overlapping dates: {sorted(list(date_overlap))[:5]}...")
            else:
                print(f"     ✅ No date overlap between train/val")
                
            # Check label distribution
            tr_labels = tr_df["Label"].value_counts()
            va_labels = va_df["Label"].value_counts()
            print(f"     Train labels: {dict(tr_labels)}")
            print(f"     Val labels:   {dict(va_labels)}")
            
            # Check for duplicate tweets (by content)
            tr_content = set(tr_df["Tweet Content"].values)
            va_content = set(va_df["Tweet Content"].values)
            content_overlap = tr_content & va_content
            if content_overlap:
                print(f"     ⚠️  CONTENT OVERLAP: {len(content_overlap)} identical tweets in train/val!")
            else:
                print(f"     ✅ No duplicate tweet content between train/val")
                
            # Check Previous Label distribution
            tr_prev = tr_df["Previous Label"].value_counts()
            va_prev = va_df["Previous Label"].value_counts()
            print(f"     Train prev labels: {dict(tr_prev)}")
            print(f"     Val   prev labels: {dict(va_prev)} (TBL-derived from market data)")
            
            # Check if we have the dreaded perfect correlation
            if len(tr_df) > 0 and len(va_df) > 0:
                # Sample a few examples to see if labels are too predictable
                sample_tr = tr_df[["Tweet Date", "Label", "Previous Label", "Close"]].head(3)
                sample_va = va_df[["Tweet Date", "Label", "Previous Label", "Close"]].head(3)
                print(f"     Train sample:")
                for _, row in sample_tr.iterrows():
                    print(f"       {row['Tweet Date'].date()} | Label: {row['Label']:<8} | Prev: {row['Previous Label']:<8} | Price: ${row['Close']:.2f}")
                print(f"     Val sample (Previous Labels from TBL market data):")
                for _, row in sample_va.iterrows():
                    print(f"       {row['Tweet Date'].date()} | Label: {row['Label']:<8} | Prev: {row['Previous Label']:<8} | Price: ${row['Close']:.2f}")
            print()

            tr_loader = self._make_loader(tr_df, shuffle=True)
            va_loader = self._make_loader(va_df, shuffle=False)

            tot_steps  = len(tr_loader) * self.epochs
            fold_scheduler = get_linear_schedule_with_warmup(
                fold_optimizer,
                num_warmup_steps=int(tot_steps * self.warmup_frac),
                num_training_steps=tot_steps,
            )

            # Prompt-tuning handles freezing automatically in __init__
            fold_model.bert_model.train()

            # Track metrics for this fold
            fold_history = {"fold": fold, "epochs": []}

            for epoch in range(self.epochs):
                loop = tqdm(tr_loader, desc=f"fold {fold} epoch {epoch+1}/{self.epochs}",
                leave=False)
                print(f"  Epoch {epoch+1}/{self.epochs}")
                train_loss = self._train_one_epoch_fold(loop, fold_model, fold_optimizer, fold_scheduler)
                
                # ── run validation with progress bar & capture preds ─────────
                val_loss, val_acc, y_true, y_pred = self._evaluate_fold(
                    va_loader, fold_model
                )

                # -----------------------------------------------------------------
                # 🧊  ONE-CHECKPOINT POLICY
                #     Keep only the *best* epoch for this fold on disk:
                #         models/ea_<ts>/fold3/      (config.json + pytorch_model.bin)
                # -----------------------------------------------------------------
                ckpt_dir = Path("models") / f"ea_{self.config.get('run_id','')}" \
                           / f"fold{fold}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                best_path = ckpt_dir / "metrics.json"
                prev_best = 9e9
                if best_path.exists():
                    prev_best = json.load(open(best_path))["val_loss"]

                if val_loss < prev_best:                     # IMPROVED ✔
                    # wipe old files
                    for f in ckpt_dir.glob("*"):
                        f.unlink()

                    fold_model.bert_model.to("cpu")
                    fold_model.bert_model.save_pretrained(ckpt_dir)
                    fold_model.tokenizer.save_pretrained(ckpt_dir)

                    json.dump({"epoch": epoch+1,
                               "val_loss": float(val_loss)},
                              open(best_path, "w"))

                    print(f"     💾  Saved new best for fold{fold} (val_loss={val_loss:.4f})")
                else:
                    print(f"     ⏩  No improvement – best remains {prev_best:.4f}")

                fold_model.bert_model.to(self.device)

                # quick confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                print(f"     Confusion matrix (rows=truth, cols=pred):\n{cm}")

                # ─── NEW: store it ────────────────────────────────
                self.cm_history.append(
                    {
                        "fold":   fold,
                        "epoch":  epoch + 1,
                        "cm_00":  int(cm[0, 0]), "cm_01": int(cm[0, 1]), "cm_02": int(cm[0, 2]),
                        "cm_10":  int(cm[1, 0]), "cm_11": int(cm[1, 1]), "cm_12": int(cm[1, 2]),
                        "cm_20":  int(cm[2, 0]), "cm_21": int(cm[2, 1]), "cm_22": int(cm[2, 2]),
                        "val_acc": round(val_acc, 4),
                        "val_loss": round(val_loss, 4)
                    }
                )

                # correlation Previous-Label ↔ Label on **val** split
                code_map = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
                corr = va_df[["Previous Label", "Label"]].replace(code_map).corr().iloc[0, 1]
                print(f"     Corr(PrevLabel , Label): {corr:.3f}")

                # randomised-labels sanity baseline (use global `np`)
                rand_acc = (np.random.permutation(y_true) == y_true).mean()
                print(f"     Random-baseline acc      : {rand_acc:.3f}")

                # gap printout
                print(f"     Train vs Val loss gap    : {(val_loss - train_loss):.4f}")
                
                epoch_metrics = {
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }
                fold_history["epochs"].append(epoch_metrics)
                
                # Clear, formatted metrics display
                print(f"\n  📊 FOLD {fold} EPOCH {epoch+1} RESULTS:")
                print(f"     Train Loss: {train_loss:.4f}")
                print(f"     Val Loss:   {val_loss:.4f}")
                print(f"     Val Acc:    {val_acc:.4f} ({val_acc*100:.1f}%)")
                
                # Early stopping warnings with better formatting
                if epoch > 0:
                    prev_val_loss = fold_history["epochs"][epoch-1]["val_loss"]
                    val_loss_improvement = prev_val_loss - val_loss
                    
                    if val_loss_improvement < 0.01:  # Less than 1% improvement
                        print(f"     ⚠️  Val loss plateaued (improvement: {val_loss_improvement:.4f})")
                    elif val_loss > prev_val_loss * 1.02:  # Val loss increased by 2%
                        print(f"     🚨 Val loss increased - potential overfitting!")
                    else:
                        print(f"     ✅ Val loss improved by {val_loss_improvement:.4f}")
                
                # Performance assessment  
                if val_acc > 0.85:
                    print(f"     🚨 WARNING: Very high accuracy ({val_acc*100:.1f}%) - possible label leakage!")
                elif val_acc > 0.70:
                    print(f"     ✅ Good performance ({val_acc*100:.1f}%)")
                elif val_acc > 0.50:
                    print(f"     📈 Reasonable performance ({val_acc*100:.1f}%)")
                else:
                    print(f"     📉 Low performance ({val_acc*100:.1f}%) - may need more training")
                
                # 🔍 DEBUG: Check for suspicious loss patterns
                if train_loss < 0.3:
                    print(f"     🚨 VERY LOW TRAIN LOSS ({train_loss:.4f}) - likely overfitting or data leakage!")
                if val_loss < 0.3:
                    print(f"     🚨 VERY LOW VAL LOSS ({val_loss:.4f}) - likely data leakage!")
                if abs(train_loss - val_loss) < 0.05:
                    print(f"     🚨 TRAIN/VAL LOSS TOO SIMILAR ({train_loss:.4f} vs {val_loss:.4f}) - suspicious!")
                    
                print()
                
            fold_metrics.append(fold_history)
            
            # Move model to CPU to free MPS memory for next fold
            fold_model.bert_model.to("cpu")
            print(f"  💾 Moved Fold {fold} model to CPU to free MPS memory")
            
            # Store this fold's model (now on CPU)
            self.fold_states.append({"model": fold_model})
            
        print(f"\n✅ Training complete! {len(self.fold_states)} independent fold models created.")

        # ── persist all epoch metrics for plotting ───────────────────────────
        metrics_df = (
            pd.json_normalize(
                fold_metrics,
                record_path=["epochs"],
                meta=["fold"]
            )
        )
        metrics_df.to_csv("epoch_metrics.csv", index=False)
        print("📊 Per-epoch metrics saved ➟ epoch_metrics.csv")
        
        # Save training metrics (json already imported at top – don't shadow!)
        with open("training_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)
        print("📊 Training metrics saved to training_metrics.json")

        # ─── NEW: persist confusion-matrix history ────────────────
        with open("confusion_matrices.json", "w") as f:
            json.dump(self.cm_history, f, indent=2)

        with open("confusion_matrices.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.cm_history[0].keys())
            writer.writeheader(); writer.writerows(self.cm_history)
        print("📊 Confusion-matrices saved ➟ confusion_matrices.{json,csv}")
        
        if hasattr(self, "fold_states"):
            pred_df = cross_val_predict(self, data)
            pred_df.to_csv("signals_per_tweet.csv", index=False)
            print(f"✅ wrote {len(pred_df):,} rows → signals_per_tweet.csv")

    # ------------------------------------------------------------------  
    # Helpers  
    # ------------------------------------------------------------------
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for fold-wise labeling (no global labeling to prevent leakage)."""
        # DO NOT LABEL GLOBALLY - this would cause massive data leakage
        # Labeling will be done per-fold using fit_and_label() and apply_labels()
        
        print("Preparing data for time-series CV (no global labeling)...")
        
        # Ensure we have the right column names
        if "date" in df.columns and "Tweet Date" not in df.columns:
            df = df.rename(columns={"date": "Tweet Date"})
        
        # Sort by date for TimeSeriesSplit (crucial for temporal ordering)
        data = df.copy().sort_values("Tweet Date").reset_index(drop=True)
        return data





    def _make_loader(self, df: pd.DataFrame, *, shuffle: bool) -> DataLoader:
        feats, lbls = [], []
        for _, row in df.iterrows():
            tok = self.model.preprocess_input(
                tweet_content=row["Tweet Content"],
                rsi=row["RSI"],
                roc=row["ROC"],
                previous_label=row["Previous Label"],
            )
            # Remove token_type_ids and keep other tensors as is
            if "token_type_ids" in tok:
                del tok["token_type_ids"]
            feats.append(tok)
            lbls.append(
                0 if row["Label"] == "Bearish"
                else 1 if row["Label"] == "Neutral" else 2
            )

        class _DS(Dataset):
            def __len__(self):  return len(lbls)
            def __getitem__(self, i):
                item = {k: v[0] for k, v in feats[i].items()}  # Remove batch dimension
                item["labels"] = torch.tensor(lbls[i])
                return item

        # on CUDA: multi‐worker + pin_memory; on MPS/CPU: single‐worker no pin_memory
        num_workers = 4 if self.device.type == "cuda" else 0
        pin_memory  = True if self.device.type == "cuda" else False
        
        # -----------------------------------------------------------------
        # (TRAIN)   Balanced sampling to fight Neutral domination
        # (VAL/TEST) keep natural order & skew
        # -----------------------------------------------------------------
        if shuffle:
            y = np.array(lbls)                       # 0-Bear,1-Neut,2-Bull
            freq = np.bincount(y, minlength=3)
            alpha = 0.5                              # 0=no rebalance, 1=full
            w_cls = (1.0 / np.maximum(freq, 1)) ** alpha
            weights = w_cls[y]

            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(y),                  # epoch length unchanged
                replacement=True,
            )

            return DataLoader(
                _DS(),
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        # ── validation loader ────────────────────────────────────────────
        return DataLoader(
            _DS(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )



    def _train_one_epoch_fold(self, loader: DataLoader, model: Model, optimizer: AdamW, scheduler: Any) -> float:
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(loader):
            # send all tensors to MPS/CPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if step == 0:                    # print once per epoch
                print("vocab_size:", model.tokenizer.vocab_size)
                print(" BATCH KEYS:", batch.keys())
                print("  input_ids  max id:", batch["input_ids"].max().item())
                print("  input_ids  shape:", batch["input_ids"].shape)
                # If you have an attention_mask, print its dtype & shape:
                if "attention_mask" in batch:
                    print("  attn_mask shape:", batch["attention_mask"].shape)
                
                # 🔍 DEBUG: Model now uses pure text prompts (no numeric_features)
                print("  Using pure text prompts - no separate numeric features")
                
                # Check label distribution in this batch
                batch_labels = batch["labels"].cpu().numpy()
                unique_labels, counts = np.unique(batch_labels, return_counts=True)
                print("  batch label distribution:", dict(zip(unique_labels, counts)))
            lbls  = batch.pop("labels")
            
            # Filter out arguments that model expects - only pass input_ids and attention_mask
            model_args = {k: v for k, v in batch.items() 
                         if k in ["input_ids", "attention_mask"]}
            
            # Use model's forward method (handles prompt-tuning automatically)
            logits = model.forward(model_args)
            
            # cost matrix so flips → 2 × penalty (was 3), Neutral errors → 1 ×
            cost = torch.tensor([[0,1,2],
                                [1,0,1],
                                [2,1,0]], device=logits.device, dtype=logits.dtype)
            # compute expected cost
            probs = torch.softmax(logits, dim=-1)
            ce     = torch.nn.functional.cross_entropy(logits, lbls, reduction='none')
            flip_c = cost[lbls, probs.argmax(dim=-1)]
            loss   = (ce * (1 + flip_c)).mean()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            loss.backward()
            optimizer.step();  scheduler.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                print(f"    step {step:<4}  train_loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / max(num_batches, 1)
        return avg_train_loss

    def _evaluate_fold(
        self, loader: DataLoader, model: Model
    ) -> Tuple[float, float, list[int], list[int]]:
        model.bert_model.eval()
        correct = total = 0
        total_loss = 0.0
        num_batches = 0
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="eval", leave=False):
                # move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                lbls = batch.pop("labels")
                
                # Filter out arguments that model expects - only pass input_ids and attention_mask
                model_args = {k: v for k, v in batch.items() 
                             if k in ["input_ids", "attention_mask"]}
                
                logits = model.forward(model_args)
                probs  = torch.softmax(logits, dim=-1)      # [B,3]

                # ── confidence gating ─────────────────────
                pred_idx = probs.argmax(dim=-1)             # [B]
                # grab the probability of the chosen class for every item
                conf = probs.gather(1, pred_idx.unsqueeze(1)).squeeze(1)  # [B]
                low_conf = conf < 0.35
                pred_idx[low_conf] = 1                      # force to Neutral
                preds = pred_idx
                
                # Calculate loss and accuracy
                loss = torch.nn.functional.cross_entropy(logits, lbls)
                total_loss += loss.item()
                num_batches += 1
                
                y_true.extend(lbls.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        
        avg_loss = total_loss / max(num_batches, 1)
        acc = correct / max(total, 1)
        
        model.bert_model.train()
        return avg_loss, acc, y_true, y_pred

def cross_val_predict(
    trainer: "Trainer",
    data: pd.DataFrame,
    cfg_path: str = "config.yaml",
    gap_days: int = 13,
) -> pd.DataFrame:
    """
    Run the already-trained fold models on *their own* validation slices and
    return, for every tweet row that was ever in a val split:
        • Pred_Label  – majority vote of the folds that predicted it
        • Pred_Conf   – mean softmax prob of that majority label
    Rows that never fell into a validation window keep NaNs.
    """
    # ---------- reproduce the blocked-purged day splits --------------------
    data = trainer._prepare_data(data)              # ensures date sort
    unique_days = (
        data["Tweet Date"].dt.normalize()
            .sort_values().drop_duplicates()
            .reset_index(drop=True)
    )
    n_days   = len(unique_days)
    fold_len = n_days // 5

    day_lookup = pd.Series(
        np.arange(n_days, dtype=int),
        index=unique_days.values            # Timestamp → 0-based int
    )
    row_day_idx = day_lookup[data["Tweet Date"].dt.normalize()].values

    # ---------- helpers ----------------------------------------------------
    votes = defaultdict(list)          # row-idx → list[int label_id]
    confs = defaultdict(list)          # row-idx → list[float prob]

    from preprocessor import Preprocessor
    from market_labeler_ewma import MarketLabelerTBL, MarketFeatureGenerator

    device = trainer.device

    start = 0
    for fold_id, fold_state in enumerate(trainer.fold_states, 1):
        stop = start + fold_len if fold_id < 5 else n_days
        val_days = np.arange(start, stop, dtype=int)

        purge_days = {
            d for v in val_days
            for d in range(max(0, v - gap_days),
                           min(n_days, v + gap_days + 1))
        }
        train_rows = np.where(~np.isin(row_day_idx,
                                       list(val_days) + list(purge_days)))[0]
        gap_rows   = np.where( np.isin(row_day_idx, purge_days))[0]
        val_rows   = np.where( np.isin(row_day_idx,  val_days))[0]

        if not len(val_rows):
            start = stop;  continue      # safety – should not happen

        # ---------- fold-specific preprocessing & labelling -------------
        fold_pre = Preprocessor(cfg_path)
        tr_df  = fold_pre.fit_transform(data.iloc[train_rows].copy())
        # gap slice may be empty ─→ only transform if we have rows
        if len(gap_rows):
            gap_df = fold_pre.transform(data.iloc[gap_rows].copy())
        else:                               # keep a typed, empty frame
            gap_df = pd.DataFrame(columns=data.columns)
        va_df  = fold_pre.transform   (data.iloc[val_rows].copy())

        # ── 1  label training / validation with exactly the same fitter ──
        fold_lbl = MarketLabelerTBL(cfg_path)
        tr_df  = fold_lbl.fit_and_label(tr_df)   # KEEP the labelled copy
        if not gap_df.empty:
            gap_df = fold_lbl.apply_labels(gap_df)
        va_df  = fold_lbl.apply_labels(va_df)

        # ── 2  causal Previous-Label feature (needs the "Label" col) ─────
        if not gap_df.empty:
            feat_gen = MarketFeatureGenerator().fit(
                pd.concat([tr_df, gap_df], ignore_index=True)
            )
            gap_df["Previous Label"] = feat_gen.transform(gap_df)
        else:
            feat_gen = MarketFeatureGenerator().fit(tr_df)
        tr_df["Previous Label"] = feat_gen.transform(tr_df)   # optional – tidy
        va_df["Previous Label"] = feat_gen.transform(va_df)

        # ---------- inference -------------------------------------------
        model = fold_state["model"]
        model.bert_model.to(device)
        model.bert_model.eval()

        with torch.no_grad():
            for ridx, row in zip(val_rows,
                                 va_df.itertuples(index=False, name=None)):
                # ------------------------------------------------------------------
                # column order in .itertuples(name=None) is exactly df.columns
                # so we can pick by position instead of worrying about the space/underscore
                # ------------------------------------------------------------------
                tweet_txt      = row[va_df.columns.get_loc("Tweet Content")]
                rsi_val        = row[va_df.columns.get_loc("RSI")]
                roc_val        = row[va_df.columns.get_loc("ROC")]
                prev_lbl       = row[va_df.columns.get_loc("Previous Label")]

                tok = model.preprocess_input(
                    tweet_content=tweet_txt,
                    rsi=rsi_val,
                    roc=roc_val,
                    previous_label=prev_lbl,
                )
                tok.pop("token_type_ids", None)
                tok = {k: v.to(device) for k, v in tok.items()}

                logits = model.forward({k: v for k, v in tok.items()
                                        if k in ("input_ids", "attention_mask")})
                probs = torch.softmax(logits, dim=-1)[0]
                
                # Confidence gating
                pred_idx = probs.argmax().item()
                conf = probs[pred_idx].item()
                # Apply confidence threshold - if confidence < 0.35, predict Neutral
                if conf < 0.35:
                    pred_idx = 1          # Neutral
                
                pred = pred_idx

                votes[ridx].append(pred)
                confs[ridx].append(probs[pred].item())

        model.bert_model.to("cpu")
        start = stop

    # ---------- assemble output -------------------------------------------
    out = data.copy()
    out["Pred_Label"] = pd.NA
    out["Pred_Conf"]  = np.nan

    if votes:
        maj = {i: max(set(v), key=v.count) for i, v in votes.items()}
        cnf = {i: np.mean(confs[i])       for i in votes.keys()}

        out.loc[list(maj.keys()), "Pred_Label"] = pd.Series(maj).map({
            0: "Bearish", 1: "Neutral", 2: "Bullish"
        })
        out.loc[list(cnf.keys()), "Pred_Conf"]  = pd.Series(cnf)

    return out