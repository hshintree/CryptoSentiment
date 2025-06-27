#!/usr/bin/env python3
"""
SingleTrainer – train ONCE on EA (#1train.csv) and evaluate on:
   • EA itself   (in-sample diagnostic)
   • EB (#2val)  (true out-of-sample test)

No folds, no blocked-purged logic, everything else (pre-processing,
EWMA-TBL labelling, causal Previous-Label, confidence gating, cost-matrix
loss) re-uses the components you already wrote.
"""
import yaml, json, csv, os, torch
from pathlib import Path
from typing import Tuple, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, \
                             precision_recall_fscore_support

from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch import nn

from model                 import Model
from preprocessor          import Preprocessor
from market_labeler_ewma   import MarketLabelerTBL, MarketFeatureGenerator

# --------------------------------------------------------------------- #
#                         ─── Helper functions ───                       #
# --------------------------------------------------------------------- #
def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    # 1) rename only if we don't already have a 'Tweet Date'
    if "date" in df.columns and "Tweet Date" not in df.columns:
        df = df.rename(columns={"date": "Tweet Date"})
    # 2) if both exist, drop the old 'date' to avoid duplicates
    elif "date" in df.columns and "Tweet Date" in df.columns:
        df = df.drop(columns=["date"])

    # 3) final safeguard – remove *any* accidental dup-names
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # guarantee dtype
    df["Tweet Date"] = pd.to_datetime(df["Tweet Date"], errors="coerce")
    return df

# --------------------------------------------------------------------- #
#                         ─── Trainer class ───                         #
# --------------------------------------------------------------------- #
class SingleTrainer:
    def __init__(self,
                 model: Model,
                 cfg_path: str = "config.yaml",
                 device: str  | None = None) -> None:

        self.cfg_path = cfg_path
        self.config   = yaml.safe_load(open(cfg_path))

        # device ----------------------------------------------------------------
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = model
        self.model.device = self.device
        self.model.bert_model.to(self.device)

        if self.device.type == "mps":                 # MPS embedding fix
            emb = self.model.bert_model.get_input_embeddings()
            emb.weight = nn.Parameter(emb.weight.to(self.device))

        # training hp – reuse values from YAML ----------------------------------
        tcfg               = self.config["training"]
        self.lr            = float(tcfg.get("learning_rate", 2e-5))
        self.epochs        = int  (tcfg.get("epochs",        5))
        self.batch_size    = int  (tcfg.get("batch_size",   12))
        self.warmup_frac   = float(tcfg.get("warmup_steps", 0.2))

        # reproducibility -------------------------------------------------------
        seed = tcfg.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # optimiser – **all trainable params**
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.bert_model.parameters()),
            lr=self.lr
        )

        self.history: list[dict] = []

    # ------------------------------------------------------------------ #
    #                       Data pipeline helpers                         #
    # ------------------------------------------------------------------ #
    def _prepare(self, df: pd.DataFrame,
                 *, fit_preproc_on: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Full pipeline:  Preprocess → EWMA-TBL label → causal PrevLabel.
        `fit_preproc_on` allows us to fit scalers ONLY on EA, then
        re-use them for EB.
        """
        # ---------- 1  preprocessing -----------------------------------------
        pre = Preprocessor(self.cfg_path)
        if fit_preproc_on is None:              # training set
            df_prep = pre.fit_transform(df.copy())
            self._pre = pre                     # cache for later
        else:                                   # EB / diagnostics
            pre.scaler = self._pre.scaler       # copy fitted scaler
            df_prep = pre.transform(df.copy())

        # ---------- 2  EWMA-TBL labelling ------------------------------------
        lbl = MarketLabelerTBL(self.cfg_path)
        if fit_preproc_on is None:              # fit on EA only
            df_lab = lbl.fit_and_label(df_prep)
            self._labeler = lbl
        else:
            df_lab = self._labeler.apply_labels(df_prep)

        # ---------- 3  causal Previous-Label ---------------------------------
        gen = MarketFeatureGenerator()
        if fit_preproc_on is None:
            df_lab["Previous Label"] = gen.transform(df_lab)  # shift-only
            self._feat_gen = gen
        else:
            df_lab["Previous Label"] = gen.transform(df_lab)

        return df_lab

    def _make_loader(self, df: pd.DataFrame,
                     *, shuffle: bool) -> DataLoader:
        feats, lbls = [], []
        for _, row in df.iterrows():
            tok = self.model.preprocess_input(
                tweet_content = row["Tweet Content"],
                rsi           = row["RSI"],
                roc           = row["ROC"],
                previous_label= row["Previous Label"],
            )
            tok.pop("token_type_ids", None)
            feats.append(tok)
            lbls.append({"Bearish":0,"Neutral":1,"Bullish":2}[row["Label"]])

        class _DS(Dataset):
            def __len__(self):  return len(lbls)
            def __getitem__(self, i):
                d = {k: v[0] for k,v in feats[i].items()}
                d["labels"] = torch.tensor(lbls[i])
                return d

        return DataLoader(_DS(), batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=0,
                          pin_memory=False)

    # ------------------------------------------------------------------ #
    #                              train                                 #
    # ------------------------------------------------------------------ #
    def fit(self, ea_df: pd.DataFrame,
            *, val_frac: float = 0.10) -> None:

        ea_df  = ea_df.sort_values("Tweet Date").reset_index(drop=True)

        # --- use full dataset for both training and validation ----------------
        train_proc = self._prepare(ea_df)   # fit preproc & labeler

        train_loader = self._make_loader(train_proc, shuffle=True)
        val_loader   = self._make_loader(train_proc, shuffle=False)  # same data, no shuffle
        
        tot_steps = len(train_loader) * self.epochs
        sched = get_linear_schedule_with_warmup(
            self.optimizer,
            int(tot_steps * self.warmup_frac),
            tot_steps
        )
        
        # ---------- choose class weights α (optional) ----------
        # Count class frequencies in training data
        labels = train_proc["Label"].values
        n_bear = (labels == "Bearish").sum()
        n_neut = (labels == "Neutral").sum()
        n_bull = (labels == "Bullish").sum()
        
        freq = torch.tensor([n_bear, n_neut, n_bull], dtype=torch.float32)
        alpha = (1.0 / freq) * (freq.sum() / 3)         # inversely proportional
        alpha = alpha.to(self.device)

        gamma = 2.0

        def focal_loss(logits, targets, alpha, gamma):
            ce = torch.nn.functional.cross_entropy(
                    logits, targets, reduction='none')      # per-sample CE
            p_t = torch.exp(-ce)                           # = softmax prob of true class
            foc = (alpha[targets] * (1 - p_t) ** gamma * ce).mean()
            return foc

        # cost = torch.tensor([[0,1,2],
        #                      [1,0,1],
        #                      [2,1,0]], device=self.device, dtype=torch.float32)

        self.model.bert_model.train()
        for epoch in range(1, self.epochs + 1):
            loop = tqdm(train_loader, desc=f"epoch {epoch}/{self.epochs}", leave=False)
            running = 0
            # --------------- training loop (unchanged) ----------------------
            for step, batch in enumerate(loop, 1):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                lbls  = batch.pop("labels")
                logits= self.model.forward({k:v for k,v in batch.items()
                                            if k in ("input_ids","attention_mask")})

                # ----- training loop -----
                loss = focal_loss(logits, lbls, alpha, gamma)

                loss.backward()
                self.optimizer.step();  sched.step();  self.optimizer.zero_grad()
                running += loss.item()

                if step % 100 == 0:
                    loop.set_postfix(loss=running/step)
            
            train_loss = running / step

            # ----------- quick eval on full training set --------------------
            val_true, val_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k:v.to(self.device) for k,v in batch.items()}
                    lbls  = batch.pop("labels")
                    logits= self.model.forward({k:v for k,v in batch.items()
                                                if k in ("input_ids","attention_mask")})
                    val_true.extend(lbls.cpu().tolist())
                    val_pred.extend(logits.argmax(dim=-1).cpu().tolist())

            prec, rec, f1, _ = precision_recall_fscore_support(
                val_true, val_pred, average="macro", zero_division=0)

            # ---- store & print summary ------------------------------------
            self.history.append({"epoch": epoch,
                                 "train_loss": train_loss,
                                 "val_f1": f1,
                                 "val_prec": prec,
                                 "val_rec": rec})

            print(f"Epoch {epoch:2d} | loss {train_loss:6.4f} | "
                  f"train-F1 {f1:.3f}  prec {prec:.3f}  rec {rec:.3f}")

    # ------------------------------------------------------------------ #
    #                              eval                                  #
    # ------------------------------------------------------------------ #
    def evaluate(self, df: pd.DataFrame,
                 *, name: str) -> dict[str,Any]:
        df_proc = self._prepare(df, fit_preproc_on=self._pre)
        loader  = self._make_loader(df_proc, shuffle=False)

        self.model.bert_model.eval()
        y_true: list[int] = []; y_pred: list[int] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"eval {name}", leave=False):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                lbls  = batch.pop("labels")
                logits= self.model.forward({k:v for k,v in batch.items()
                                            if k in ("input_ids","attention_mask")})
                probs = torch.softmax(logits, dim=-1)
                pred  = probs.argmax(dim=-1).cpu().tolist()

                # confidence gating
                # conf  = probs.max(dim=-1).values
                # pred  = [p if c>=0.45 else 1 for p,c in zip(pred,conf.cpu())]

                y_true.extend(lbls.cpu().tolist())
                y_pred.extend(pred)

        cm  = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        acc = accuracy_score(y_true, y_pred)
        prec,rec,f1,_ = precision_recall_fscore_support(
            y_true,y_pred,average="macro",zero_division=0)

        print(f"\n{name} confusion-matrix (rows=truth 0/1/2, cols=pred):\n{cm}")
        print(f"{name} – acc {acc:.3f}  prec {prec:.3f}  rec {rec:.3f}  f1 {f1:.3f}")

        return {"name":name,"acc":acc,"prec":prec,"rec":rec,"f1":f1,
                "cm":cm.tolist()}

# --------------------------------------------------------------------- #
#                        tiny "export" function                         #
# --------------------------------------------------------------------- #
def save_model(model: Model, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    model.bert_model.save_pretrained(outdir)
    model.tokenizer.save_pretrained(outdir)
    print(f"✓ model saved to {outdir}")

def plot_training_history(trainer: SingleTrainer, outdir: Path | None = None):
    """
    Plot training history showing loss and validation metrics.
    
    Args:
        trainer: Trained SingleTrainer instance with history
        outdir: Optional directory to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        if not trainer.history:
            print("No training history to plot")
            return
            
        df = pd.DataFrame(trainer.history)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Training Loss
        ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Validation Metrics
        ax2.plot(df['epoch'], df['val_f1'], 'g-', linewidth=2, label='F1-Score')
        ax2.plot(df['epoch'], df['val_prec'], 'r-', linewidth=2, label='Precision')
        ax2.plot(df['epoch'], df['val_rec'], 'orange', linewidth=2, label='Recall')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Validation Metrics Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if outdir:
            plot_path = outdir / "training_history.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)     # ▲ new
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history plot saved to {plot_path}")
        
        plt.show()
        
    except ImportError:
        print("matplotlib not available - skipping plot generation")
        # Fallback: print summary statistics
        if trainer.history:
            df = pd.DataFrame(trainer.history)
            print("\nTraining Summary:")
            print(f"Final train loss: {df['train_loss'].iloc[-1]:.4f}")
            print(f"Best val F1: {df['val_f1'].max():.3f} (epoch {df['val_f1'].idxmax()+1})")
            print(f"Final val F1: {df['val_f1'].iloc[-1]:.3f}")

# --------------------------------------------------------------------- #
#                           Usage Example                                #
# --------------------------------------------------------------------- #
"""
Example usage:

# Load and prepare data
ea_raw = pd.read_csv("data/#1train.csv")
eb_raw = pd.read_csv("data/#2val.csv")

# Apply date coercion
ea_df = _coerce_dates(ea_raw)
eb_df = _coerce_dates(eb_raw)

# Train and evaluate
model = Model(config["model"])
trainer = SingleTrainer(model, "config.yaml")

# Train with per-epoch validation metrics
trainer.fit(ea_df, val_frac=0.10)  # 10% validation slice

# Plot training history
plot_training_history(trainer, outdir=Path("models/single_train"))

# Evaluate on both datasets
ea_results = trainer.evaluate(ea_df, name="EA")
eb_results = trainer.evaluate(eb_df, name="EB")

# Save the trained model
save_model(model, Path("models/single_train"))
"""
