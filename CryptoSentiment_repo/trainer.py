# trainer.py  — hot-fix: ensure numeric hyper-parameters are cast to float/int
import yaml
import torch
from torch import nn
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding
try:
    from transformers.optimization import AdamW          # HF ≥4.40
except ImportError:
    from torch.optim import AdamW                        # fallback

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle

from model                 import Model
from market_labeler_ewma   import MarketLabelerEWMA


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
        # MPS doesn’t yet support scaled_dot_product_attention — use CPU (or CUDA)
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # move the model once
        self.model.bert_model.to(self.device)
        # (if you have any other submodules holding parameters, move them too)
        
        self.fold_states = []

        tcfg = self.config["training"]
        # ---- cast YAML scalars safely -----------------------------------
        self.learning_rate = float(tcfg.get("learning_rate", 1e-5))
        self.batch_size    = int  (tcfg.get("batch_size",    12))
        self.epochs        = int  (tcfg.get("epochs",        2))
        self.warmup_frac   = float(tcfg.get("warmup_steps",  0.1))
        # -----------------------------------------------------------------

        self.optimizer = AdamW(self.model.bert_model.parameters(),
                               lr=self.learning_rate)
        self.scheduler = None

    # ------------------------------------------------------------------  
    # Public API  
    # ------------------------------------------------------------------
    def train(self) -> None:
        """5-fold grouped CV + standard training loop."""
        data = self._prepare_data(self.data)

        gkf = GroupKFold(n_splits=5)
        
        # Store original model config for reinitializing
        original_model_cfg = self.model
        
        # ✅ TRAINING METRICS TRACKING
        fold_metrics = []
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(data,
                                                         groups=data["group"])):
            print(f"\n── Fold {fold+1}/{gkf.n_splits} ──")
            
            # ✅ VALIDATE NO TRAIN/VAL OVERLAP
            train_set = set(tr_idx)
            val_set = set(va_idx)
            assert not (train_set & val_set), f"❌ Fold {fold+1}: train/val overlap detected!"
            print(f"✅ Fold {fold+1}: No train/val overlap (train={len(train_set)}, val={len(val_set)})")
            
            # ✅ REINITIALIZE MODEL & OPTIMIZER FOR EACH FOLD
            print(f"🔄 Fold {fold+1}: Reinitializing model and optimizer...")
            
            # Create fresh model for this fold
            fold_model = Model(self.config["model"])
            fold_model.bert_model.to(self.device)
            
            # Create fresh optimizer for this fold - only trainable parameters
            if hasattr(fold_model, 'use_prompt_tuning') and fold_model.use_prompt_tuning:
                # Only optimize prompt embeddings + classifier for prompt-tuning
                trainable_params = list(fold_model.bert_model.classifier.parameters())
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
            
            # ✅ FOLD-WISE LABELING: Fit on training data, apply to both train/val
            print(f"  🔬 Applying fold-wise labeling to prevent EWMA threshold leakage...")
            fold_labeler = MarketLabelerEWMA("config_ewma.yaml")
            
            # Fit labeler on training data and apply labels
            tr_df = fold_labeler.fit_and_label(tr_df)
            # Apply same fitted thresholds to validation data  
            va_df = fold_labeler.apply_labels(va_df)
            
            # Recompute "Previous Label" separately for each split to prevent leakage
            tr_df["Previous Label"] = tr_df["Label"].shift(1).fillna("Neutral") 
            va_df["Previous Label"] = va_df["Label"].shift(1).fillna("Neutral")
            
            # 🔍 DEBUG: Analyze data leakage potential
            print(f"\n  🔍 FOLD {fold+1} DEBUG ANALYSIS:")
            
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
            print(f"     Val prev labels:   {dict(va_prev)}")
            
            # Check if we have the dreaded perfect correlation
            if len(tr_df) > 0 and len(va_df) > 0:
                # Sample a few examples to see if labels are too predictable
                sample_tr = tr_df[["Tweet Date", "Label", "Previous Label", "Close"]].head(3)
                sample_va = va_df[["Tweet Date", "Label", "Previous Label", "Close"]].head(3)
                print(f"     Train sample:")
                for _, row in sample_tr.iterrows():
                    print(f"       {row['Tweet Date'].date()} | Label: {row['Label']:<8} | Prev: {row['Previous Label']:<8} | Price: ${row['Close']:.2f}")
                print(f"     Val sample:")
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
            fold_history = {"fold": fold+1, "epochs": []}

            for epoch in range(self.epochs):
                loop = tqdm(tr_loader, desc=f"fold {fold+1} epoch {epoch+1}/{self.epochs}",
                            leave=False)
                print(f"  Epoch {epoch+1}/{self.epochs}")
                train_loss = self._train_one_epoch_fold(loop, fold_model, fold_optimizer, fold_scheduler)
                val_loss, val_acc = self._evaluate_fold(va_loader, fold_model)
                
                epoch_metrics = {
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }
                fold_history["epochs"].append(epoch_metrics)
                
                # Clear, formatted metrics display
                print(f"\n  📊 FOLD {fold+1} EPOCH {epoch+1} RESULTS:")
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
            print(f"  💾 Moved Fold {fold+1} model to CPU to free MPS memory")
            
            # Store this fold's model (now on CPU)
            self.fold_states.append({"model": fold_model})
            
        print(f"\n✅ Training complete! {len(self.fold_states)} independent fold models created.")
        
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
        
        print("Preparing data for fold-wise labeling (no global labeling)...")
        
        # Ensure we have the right column names
        if "date" in df.columns and "Tweet Date" not in df.columns:
            df = df.rename(columns={"date": "Tweet Date"})
        
        # Only assign groups, don't label yet
        data = df.copy()
        data["group"] = self._assign_groups(data)
        return data

    @staticmethod
    def _assign_groups(df: pd.DataFrame) -> np.ndarray:
        """Temporal grouping to prevent data leakage - group by date periods."""
        # Get unique dates and sort them
        unique_dates = sorted(df['Tweet Date'].dt.date.unique())
        n_dates = len(unique_dates)
        
        # Divide dates into 5 temporal groups
        dates_per_group = n_dates // 5
        
        # Create date-to-group mapping
        date_to_group = {}
        for i in range(5):
            start_date_idx = i * dates_per_group
            end_date_idx = (i + 1) * dates_per_group if i < 4 else n_dates
            
            for date_idx in range(start_date_idx, end_date_idx):
                date_to_group[unique_dates[date_idx]] = i
        
        # Assign groups based on tweet dates
        groups = df['Tweet Date'].dt.date.map(date_to_group).values
        
        return groups


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

        return DataLoader(
            _DS(),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0 if self.device.type == 'mps' else 0,  # MPS doesn't pickle well, use 0
            pin_memory=False,  # Disable pin_memory as it's not supported on MPS
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
            
            # Filter out arguments that model expects
            model_args = {k: v for k, v in batch.items() 
                         if k in ["input_ids", "attention_mask", "token_type_ids"]}
            
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

    def _evaluate_fold(self, loader: DataLoader, model: Model) -> Tuple[float, float]:
        model.bert_model.eval()
        correct = total = 0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                # move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                lbls = batch.pop("labels")
                
                # Filter out arguments that model expects
                model_args = {k: v for k, v in batch.items() 
                             if k in ["input_ids", "attention_mask", "token_type_ids"]}
                
                logits = model.forward(model_args)
                preds = logits.argmax(dim=-1)
                
                # Calculate loss and accuracy
                loss = torch.nn.functional.cross_entropy(logits, lbls)
                total_loss += loss.item()
                num_batches += 1
                
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        
        avg_loss = total_loss / max(num_batches, 1)
        acc = correct / max(total, 1)
        
        model.bert_model.train()
        return avg_loss, acc

def cross_val_predict(trainer: "Trainer",
                      data: pd.DataFrame,
                      cfg_path: str = "config.yaml") -> pd.DataFrame:
    """
    Run the already-trained 5 fold models on validation data only and return
    majority-vote label + confidence (mean softmax prob of winning class).
    FIXED: Uses model's own predictions for Previous Label to prevent leakage.
    """
    from collections import defaultdict
    from sklearn.model_selection import GroupKFold
    
    votes   = defaultdict(list)   # idx → [class_ids]
    confs   = defaultdict(list)   # idx → [prob_of_pred]

    # ensure inference uses the same device
    device = trainer.device
    
    # Get the same grouping as training
    data_grouped = trainer._prepare_data(data)
    gkf = GroupKFold(n_splits=5)
    
    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(data_grouped, groups=data_grouped["group"])):
        if fold_id >= len(trainer.fold_states):
            break
            
        model = trainer.fold_states[fold_id]["model"]
        
        # Get validation data for this fold
        va_df = data_grouped.iloc[va_idx].copy()
        
        # Process and label validation data (same as training)
        from preprocessor import Preprocessor
        fold_preprocessor = Preprocessor(trainer.config_path)
        va_processed = fold_preprocessor.fit_transform(va_df)  # Use same preprocessing
        
        from market_labeler_ewma import MarketLabelerEWMA
        fold_labeler = MarketLabelerEWMA("config_ewma.yaml")
        va_labeled = fold_labeler.fit_and_label(va_processed)  # Need labels for inference
        
        # Build Previous Label using MODEL PREDICTIONS (no leakage)
        va_labeled = va_labeled.sort_values("Tweet Date").reset_index(drop=True)
        prev_labels = ["Neutral"]  # Start with neutral
        
        # Temporarily move model to device
        model.bert_model.to(device)
        model.bert_model.eval()
        
        with torch.no_grad():
            for i, row in va_labeled.iterrows():
                # Use previous MODEL prediction as Previous Label
                current_prev_label = prev_labels[-1] if prev_labels else "Neutral"
                
                # Create input with model's own previous prediction
                tok = model.preprocess_input(
                    tweet_content=row["Tweet Content"],
                    rsi=row["RSI"],
                    roc=row["ROC"],
                    previous_label=current_prev_label,
                )
                
                # Remove token_type_ids and move to device
                if "token_type_ids" in tok:
                    del tok["token_type_ids"]
                batch = {k: v.to(device) for k, v in tok.items()}
                
                # Get prediction
                model_args = {k: v for k, v in batch.items() 
                             if k in ["input_ids", "attention_mask", "token_type_ids"]}
                
                logits = model.forward(model_args)
                probs = torch.softmax(logits, dim=-1)
                pred = probs.argmax(dim=-1).item()
                
                # Store vote
                original_idx = va_idx[i]
                votes[original_idx].append(pred)
                confs[original_idx].append(probs[0, pred].item())
                
                # Update previous labels with this prediction
                pred_label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}[pred]
                prev_labels.append(pred_label)
        
        # Move model back to CPU
        model.bert_model.to("cpu")

    # majority + avg-confidence
    maj_lbl = {i: max(set(votes[i]), key=votes[i].count) for i in votes}
    maj_cnf = {i: np.mean(confs[i]) for i in confs}

    out = data.copy()
    out["Pred_Label"]   = out.index.map(maj_lbl)
    out["Pred_Conf"]    = out.index.map(maj_cnf)
    out["Pred_Label"]   = out["Pred_Label"].map({0: "Bearish",
                                                 1: "Neutral",
                                                 2: "Bullish"})
    return out