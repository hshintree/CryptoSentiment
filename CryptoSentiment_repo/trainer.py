# trainer.py  — hot-fix: ensure numeric hyper-parameters are cast to float/int
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Any, Dict
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

from model            import Model
from market_labeler   import MarketLabeler


class Trainer:
    """Handle training of the BERT-based model with grouped CV."""

    def __init__(self,
                 model: Model,
                 data: pd.DataFrame,
                 config_path: str = "config.yaml") -> None:

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = model
        self.data  = data
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
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(data,
                                                         groups=data["group"])):
            print(f"\n── Fold {fold+1}/{gkf.n_splits} ──")
            tr_df, va_df = data.iloc[tr_idx], data.iloc[va_idx]

            tr_loader = self._make_loader(tr_df, shuffle=True)
            va_loader = self._make_loader(va_df, shuffle=False)

            tot_steps  = len(tr_loader) * self.epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(tot_steps * self.warmup_frac),
                num_training_steps=tot_steps,
            )

            self.model.freeze_layers(11)
            self.model.bert_model.train()

            for epoch in range(self.epochs):
                loop = tqdm(tr_loader, desc=f"fold {fold+1} epoch {epoch+1}/{self.epochs}",
                leave=False)
                print(f"  Epoch {epoch+1}/{self.epochs}")
                self._train_one_epoch(loop)
                self._evaluate(va_loader)
                # keep a lightweight reference for later voting
                self.fold_states.append({"model": self.model})
        if hasattr(self, "fold_states"):
            pred_df = cross_val_predict(self, data)
            pred_df.to_csv("signals_per_tweet.csv", index=False)
            print(f"✅ wrote {len(pred_df):,} rows → signals_per_tweet.csv")

    # ------------------------------------------------------------------  
    # Helpers  
    # ------------------------------------------------------------------
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add TBL labels, Previous-Label feature and CV groups."""
        lbl = MarketLabeler().label_data(df)
        lbl["Previous Label"] = lbl["Label"].shift(1).fillna("Neutral")
        lbl["group"]          = self._assign_groups(lbl)
        return lbl

    @staticmethod
    def _assign_groups(df: pd.DataFrame) -> np.ndarray:
        """Random 20 % time buckets so tweets close in time stay together."""
        return shuffle(df.index.to_numpy()) // max(len(df) // 5, 1)

# ------------------------------------------------------------------ #
# inside class Trainer
# ------------------------------------------------------------------ #
    def _make_loader(self, df: pd.DataFrame, *, shuffle: bool) -> DataLoader:
        feats, lbls = [], []
        for _, row in df.iterrows():
            feats.append(
                self.model.preprocess_input(
                    tweet_content=row["Tweet Content"],
                    rsi=row["RSI"],
                    roc=row["ROC"],
                    date=row["Tweet Date"].strftime("%Y-%m-%d"),
                    previous_label=row["Previous Label"],
                )
            )
            lbls.append(
                0 if row["Label"] == "Bearish"
                else 1 if row["Label"] == "Neutral" else 2
            )

        class _DS(Dataset):
            def __len__(self):  return len(lbls)
            def __getitem__(self, i):
                item = {k: torch.tensor(v) for k, v in feats[i].items()}
                item["labels"] = torch.tensor(lbls[i])
                return item

        return DataLoader(
            _DS(),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )


    def _train_one_epoch(self, loader: DataLoader) -> None:
        for step, batch in enumerate(loader):
            lbls  = batch.pop("labels")
            outs  = self.model.bert_model(**batch)
            loss  = torch.nn.functional.cross_entropy(outs.logits, lbls)
            loss.backward()
            self.optimizer.step();  self.scheduler.step()
            self.optimizer.zero_grad()
            if step == 0:                    # print once per epoch
                 print("shape :", batch["input_ids"].shape)   # e.g. torch.Size([4, 256])
            if step % 10 == 0:
                print(f"    step {step:<4}  loss {loss.item():.4f}")

    def _evaluate(self, loader: DataLoader) -> None:
        self.model.bert_model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                lbls = batch.pop("labels")
                preds = self.model.bert_model(**batch).logits.argmax(dim=-1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
        acc = correct / total if total else 0
        print(f"  → val acc: {acc:.4f}")
        self.model.bert_model.train()

def cross_val_predict(trainer: "Trainer",
                      data: pd.DataFrame,
                      cfg_path: str = "config.yaml") -> pd.DataFrame:
    """
    Run the already-trained 5 fold models on *all* data and return
    majority-vote label + confidence (mean softmax prob of winning class).
    """
    from collections import defaultdict
    votes   = defaultdict(list)   # idx → [class_ids]
    confs   = defaultdict(list)   # idx → [prob_of_pred]

    for fold_id, fold_state in enumerate(trainer.fold_states):
        model = fold_state["model"]            # reuse fine-tuned weights
        loader = trainer._make_loader(data, shuffle=False)

        model.bert_model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                idx   = batch_idx              # 1-to-1 because no shuffle
                logits = model.bert_model(**{k: v for k, v in batch.items()
                                             if k != "labels"}).logits
                probs  = torch.softmax(logits, dim=-1)
                pred   = probs.argmax(dim=-1).item()
                votes[idx].append(pred)
                confs[idx].append(probs[0, pred].item())

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