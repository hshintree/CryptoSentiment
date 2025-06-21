import os
# enable MPS→CPU fallback for any unsupported kernels
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# trainer.py  — hot-fix: ensure numeric hyper-parameters are cast to float/int
import yaml
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

from torch.utils.data import DataLoader, Dataset
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

        # ----------------  split on UNIQUE DATES  -------------------------
        gap_days = 30
        unique_dates = (
            self.data.assign(__day=self.data["Tweet Date"].dt.normalize())
                .drop_duplicates("__day")
                .sort_values("__day")
                .reset_index()          # keep original tweet-row index
                .rename(columns={"__day": "Day"})
        )

        # one row == one day  ➜ safe to use gap in DAYS
        total_days = len(unique_dates)
        test_days  = total_days // 5      # 20 % of days
        max_splits = (total_days - gap_days) // test_days
        n_splits   = min(5, max_splits)
        if n_splits < 2:
            raise ValueError(
                f"too few days ({total_days}) for the requested gap/test sizes"
            )

        date_splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_days,
            gap=gap_days,
        )
        print(f"Using Date-wise TimeSeriesSplit("
              f"n_splits={n_splits}, test_days={test_days}, gap_days={gap_days})")

        # ----- identical to market_labeler's day-level grouping ------------
        day_series  = self.data["Tweet Date"].dt.normalize()   # 00:00 timestamps
        day_to_rows = (
            self.data
                .assign(__day=day_series)
                .groupby("__day", sort=False)["__day"]
                .apply(lambda g: g.index.values)   # list of row indices per day
                .to_dict()
        )

        def expand(day_idx):
            """Convert scikit-learn's day indices → tweet-row indices."""
            days = unique_dates.loc[list(day_idx), "Day"]   # Timestamp keys
            rows = []
            for d in days:
                rows.extend(day_to_rows[d])
            return np.asarray(rows, dtype=int)
        
        # ✅ TRAINING METRICS TRACKING
        fold_metrics = []
        
        for fold, (day_tr, day_val) in enumerate(date_splitter.split(unique_dates), 1):
            tr_idx = expand(day_tr)
            va_idx = expand(day_val)

            # pretty diagnostic of the actual temporal split
            train_dates = data.iloc[tr_idx]["Tweet Date"]
            val_dates   = data.iloc[va_idx]["Tweet Date"]
            print(f"Fold {fold}: "
                  f"train ≤ {train_dates.max().date()}, "
                  f"gap = {train_dates.max().date()+pd.Timedelta(days=1)} … "
                  f"{val_dates.min().date()-pd.Timedelta(days=1)}, "
                  f"val ≥ {val_dates.min().date()}")
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
            
            tr_df, va_df = data.iloc[tr_idx].copy(), data.iloc[va_idx].copy()
            
            # ✅ FOLD-WISE PREPROCESSING: Fit scaler on training data only
            print(f"  📊 Applying fold-wise preprocessing to prevent scaling leakage...")
            from preprocessor import Preprocessor
            fold_preprocessor = Preprocessor(self.config_path)
            
            # Fit preprocessor on training data and transform both splits
            tr_df = fold_preprocessor.fit_transform(tr_df)
            va_df = fold_preprocessor.transform(va_df)
            
            # ✅ FOLD-WISE LABELING: Batch-label with TBL
            print(f"  🔬 Applying fold-wise TBL labeling…")
            fold_labeler = MarketLabelerTBL("config.yaml")
            tr_df = fold_labeler.fit_and_label(tr_df)
            va_df = fold_labeler.apply_labels(va_df)

            # ✅ FOLD-WISE FEATURE: Generate causal 'Previous Label'
            print(f"  🎯 Generating causal previous-label feature…")
            feat_gen = MarketFeatureGenerator("config.yaml")
            feat_gen.fit(tr_df)
            tr_df["Previous Label"] = feat_gen.transform(tr_df)
            va_df["Previous Label"] = feat_gen.transform(va_df)

            # ── quick leakage sanity-check ρ(PrevLabel , Label) ─────────
            _code = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
            corr = va_df.replace(_code)[["Previous Label", "Label"]].corr().iloc[0, 1]
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

            print(f"ρ_day-level = {day_corr:.3f}")   # ← should be in the ~0.25–0.30 range
            
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

                # ---------------------------------------------------------
                # 🧊  QUICK CHECKPOINT
                # ---------------------------------------------------------
                # always keep *one* epoch-checkpoint per fold on CPU
                ckpt_root = Path("tmp_ckpts") / f"fold{fold}"
                ckpt_root.mkdir(parents=True, exist_ok=True)

                # nuke the previous epoch's ckpt (if any)
                if epoch > 0:
                    old = ckpt_root / f"epoch{epoch}.pt"
                    if old.exists():
                        old.unlink()

                # store the new one – model is already on self.device, move →
                fold_model.bert_model.to("cpu")
                torch.save(
                    {"state_dict": fold_model.bert_model.state_dict(),
                     "epoch": epoch+1,
                     "val_loss": float(val_loss)},
                    ckpt_root / f"epoch{epoch+1}.pt"
                )
                # move back to device for next epoch's training pass
                fold_model.bert_model.to(self.device)

                # quick confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                print(f"     Confusion matrix (rows=truth, cols=pred):\n{cm}")

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
        
        # Save training metrics
        import json
        with open("training_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)
        print("📊 Training metrics saved to training_metrics.json")
        
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
        
        return DataLoader(
            _DS(),
            batch_size=self.batch_size,
            shuffle=shuffle,
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
            loss = torch.nn.functional.cross_entropy(logits, lbls)
            
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
                preds = logits.argmax(dim=-1)
                
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

def cross_val_predict(trainer: "Trainer",
                      data: pd.DataFrame,
                      cfg_path: str = "config.yaml") -> pd.DataFrame:
    """
    Run the already-trained 5 fold models on validation data only and return
    majority-vote label + confidence (mean softmax prob of winning class).
    Uses TBL-derived Previous Labels from market data (no model predictions).
    """
    from collections import defaultdict
    from sklearn.model_selection import TimeSeriesSplit
    
    votes   = defaultdict(list)   # idx → [class_ids]
    confs   = defaultdict(list)   # idx → [prob_of_pred]

    # ensure inference uses the same device
    device = trainer.device
    
    # Get the same temporal splits as training
    data_grouped = trainer._prepare_data(data)
    gap_days = 30
    unique_dates = (
        data.assign(__day=data["Tweet Date"].dt.normalize())
            .drop_duplicates("__day")
            .sort_values("__day")
            .reset_index()          # keep original tweet-row index
            .rename(columns={"__day": "Day"})
    )

    # one row == one day  ➜ safe to use gap in DAYS
    total_days = len(unique_dates)
    test_days  = total_days // 5      # 20 % of days
    max_splits = (total_days - gap_days) // test_days
    n_splits   = min(5, max_splits)

    # ----- same grouping trick as in train() / market_labeler -------------
    day_series  = data["Tweet Date"].dt.normalize()
    day_to_rows = (
        data.assign(__day=day_series)
            .groupby("__day", sort=False)["__day"]
            .apply(lambda g: g.index.values)
            .to_dict()
    )

    def expand(day_idx):
        """Convert scikit-learn's day indices → tweet-row indices."""
        days = unique_dates.loc[list(day_idx), "Day"]
        rows = []
        for d in days:
            rows.extend(day_to_rows[d])
        return np.asarray(rows, dtype=int)

    date_splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_days,
        gap=gap_days,
    )
    
    for fold_id, (day_tr, day_val) in enumerate(date_splitter.split(unique_dates)):
        tr_idx = expand(day_tr)
        va_idx = expand(day_val)
        if fold_id >= len(trainer.fold_states):
            break
            
        model = trainer.fold_states[fold_id]["model"]
        
        # Get validation data for this fold
        va_df = data_grouped.iloc[va_idx].copy()
        
        # ── FOLD-WISE PREPROCESSING ───────────────────────────────────
        # Fit on train split, then transform val split
        from preprocessor import Preprocessor
        fold_preprocessor = Preprocessor(trainer.config_path)
        tr_df = data_grouped.iloc[tr_idx].copy()
        tr_processed = fold_preprocessor.fit_transform(tr_df)
        va_processed = fold_preprocessor.transform(va_df)
        
        # ── FOLD-WISE LABELING ───────────────────────────────────────
        from market_labeler_ewma import MarketLabelerTBL, MarketFeatureGenerator
        fold_labeler = MarketLabelerTBL("config.yaml")
        _ = fold_labeler.fit_and_label(tr_processed)
        va_labeled = fold_labeler.apply_labels(va_processed)

        feat_gen = MarketFeatureGenerator("config.yaml")
        feat_gen.fit(tr_processed)
        va_labeled["Previous Label"] = feat_gen.transform(va_labeled)
        
        # Temporarily move model to device
        model.bert_model.to(device)
        model.bert_model.eval()
        
        # Process all validation data for this fold
        with torch.no_grad():
            for i, (_, row) in enumerate(va_labeled.iterrows()):
                original_idx = va_idx[i]
                
                # Create input with TBL-derived Previous Label
                tok = model.preprocess_input(
                    tweet_content=row["Tweet Content"],
                    rsi=row["RSI"],
                    roc=row["ROC"],
                    previous_label=row["Previous Label"],
                )
                
                # Remove token_type_ids and move to device
                if "token_type_ids" in tok:
                    del tok["token_type_ids"]
                tok = {k: v.to(device) for k, v in tok.items()}
                
                # Get prediction
                model_args = {k: v for k, v in tok.items() 
                             if k in ["input_ids", "attention_mask"]}
                
                logits = model.forward(model_args)
                probs = torch.softmax(logits, dim=-1)
                pred = probs.argmax(dim=-1).item()
                
                # Store vote for this sample
                votes[original_idx].append(pred)
                confs[original_idx].append(probs[0, pred].item())
        
        # Move model back to CPU
        model.bert_model.to("cpu")

    # ── majority + avg-confidence ─────────────────────────────────
    maj_lbl = {i: max(set(votes[i]), key=votes[i].count) for i in votes}
    maj_cnf = {i: np.mean(confs[i]) for i in confs}

    out = data.copy()
    out["Pred_Label"]   = out.index.map(maj_lbl)
    out["Pred_Conf"]    = out.index.map(maj_cnf)
    out["Pred_Label"]   = out["Pred_Label"].map({0: "Bearish",
                                                 1: "Neutral",
                                                 2: "Bullish"})
    return out