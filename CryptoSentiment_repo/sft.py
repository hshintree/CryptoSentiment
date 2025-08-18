#!/usr/bin/env python3
"""
bitcoin_sft_trainer.py – Supervised Fine‑Tuning for CryptoBERT on Bitcoin tweets
===========================================================================
Changes v2
----------
* **Removed date from the prompt** to avoid time‑context anchoring (TCA).
* Adapted to **ElKulako/cryptobert** base model (BERT‑like) – still fine for causal‑LM SFT; we wrap in an AutoModelForSequenceClassification head.
* Dataset columns expected:  
  `date,Tweet Content,Label,Previous Label,Close,Volume,…` (other columns are ignored).
* Computes **14‑period RSI** & **7‑period ROC** on the fly from the `Close` price so no extra columns are needed.  Buckets are: `oversold / neutral / overbought` and `bearish / neutral / bullish` just like the paper.
* Clean `logging` stanza (ASCII hyphen) & clarified required pip installs.

Quick install
-------------
```bash
pip install "trl>=0.8.5" "transformers>=4.42.0" peft bitsandbytes accelerate datasets pandas ta
```
(🔌 `ta` provides the RSI calculation.)

Example run
-----------
```bash
python bitcoin_sft_trainer.py \
  --model_name_or_path ElKulako/cryptobert \
  --train_file "data/#1train.csv" \
  --validation_file "data/#2val.csv" \
  --output_dir checkpoints/cryptobert-bitcoin-sft \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --packing --use_peft --lora_r 32 --lora_alpha 16
```

"""
import argparse, os, logging, warnings
from pathlib import Path
from typing import Dict, List
import sys
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm                    # NEW
from transformers import (AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
                          TrainingArguments)
import transformers           # NEW
import inspect           # NEW
import trl
from trl import SFTTrainer, SFTConfig, get_peft_config, get_quantization_config, get_kbit_device_map
from ta.momentum import RSIIndicator
from ta.momentum import ROCIndicator
from torch.utils.data import DataLoader
from transformers import default_data_collator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from peft import get_peft_model, LoraConfig

# Add the parent directory to path to import our existing modules
sys.path.append(str(Path(__file__).parent))
from preprocessor import Preprocessor
from market_labeler_ewma import MarketLabelerTBL, MarketFeatureGenerator
from model import Model
from train_one import _coerce_dates

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LABEL_MAP = {"Bearish": 0, "Neutral": 1, "Bullish": 2}

#  ★ NEW bucket helpers – match Preprocessor patch ★
def bucket_rsi(val: float) -> str:
    if pd.isna(val):   return "neutral"
    if val < 30:       return "oversold"
    if val > 70:       return "overbought"
    return "neutral"

def bucket_roc(val: float, lo: float, hi: float, sigma_val: float) -> str:
    if pd.isna(val) or pd.isna(lo) or pd.isna(hi) or pd.isna(sigma_val):
        return "neutral"
    if val <= lo - sigma_val:   return "falling fast"
    if val <  0:                return "falling"
    if val >= hi + sigma_val:   return "rising fast"
    if val >  0:                return "rising"
    return "neutral"


def build_prompt(row: pd.Series) -> str:
    """Construct the pure‑text prompt (no date)."""
    return (
        f"Previous Label: {row['Previous Label']}, "
        f"RSI: {row['RSI_bucket']}, Momentum: {row['ROC_bucket']}, "
        f"Tweet: {row['Tweet Content']}\nAssistant:"
    )


def load_and_prepare(train_file: str, val_file: str, cache_dir="cached_data") -> Dict[str, Dataset]:
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        logger.info("✅ Loading cached preprocessed data...")
        df_train_lab = pd.read_parquet(train_cache)
        df_val_lab = pd.read_parquet(val_cache)
    else:
        logger.info("📂 Loading CSVs and preprocessing…")
        df_train = _coerce_dates(pd.read_csv(train_file))
        df_val = _coerce_dates(pd.read_csv(val_file))

        pre = Preprocessor("config.yaml")
        df_train_proc = pre.fit_transform(df_train.copy())
        df_val_proc = pre.transform(df_val.copy())

        labeler = MarketLabelerTBL("config.yaml", verbose=False)
        df_train_lab = labeler.fit_and_label(df_train_proc)
        df_val_lab = labeler.apply_labels(df_val_proc)

        feat_gen = MarketFeatureGenerator()
        df_train_lab["Previous Label"] = feat_gen.transform(df_train_lab)
        df_val_lab["Previous Label"] = feat_gen.transform(df_val_lab)

        df_train_lab["text"] = df_train_lab.apply(build_prompt, axis=1)
        df_val_lab["text"] = df_val_lab.apply(build_prompt, axis=1)

        logger.info("💾 Saving preprocessed data to cache...")
        df_train_lab.to_parquet(train_cache)
        df_val_lab.to_parquet(val_cache)

    return {
        "train": Dataset.from_pandas(df_train_lab[["text", "Label"]]),
        "val": Dataset.from_pandas(df_val_lab[["text", "Label"]]),
    }


def tokenize_dataset(ds: Dataset, tokenizer) -> Dataset:
    """HF datasets.map with tqdm, single-proc to avoid pickling issues on macOS."""

    def tok_fn(ex):
        tok = tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,   # now safe (≤128)
        )
        tok["labels"] = LABEL_MAP[ex["Label"]]
        return tok

    logger.info("✂️  Tokenising…")
    return ds.map(
        tok_fn,
        remove_columns=["text", "Label"],
        num_proc=1,                       # <- safer on macOS + fixes Overflow
        desc="Tokenising",                # tqdm bar title
    )


# ────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT trainer for CryptoBERT on Bitcoin tweets")

    # Model & data args
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Training hyperparams
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--packing", action="store_true")

    # LoRA
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--prompt_tuning", action="store_true",
                        help="Activate learnable prompt embeddings (prepended tokens)")

    # Quantisation
    parser.add_argument("--load_in_4bit", action="store_true")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # ensure output directory exists BEFORE trainer tries to write there
    # ------------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer & model
    logger.info("🔧 Loading base model and tokenizer for LoRA …")

    # --- BEFORE LoRA config ---
    if args.prompt_tuning:                           # <-- new CLI flag
        wrapped = Model("config.yaml")               # creates prompt_embeddings
        base_model = wrapped.bert_model              # use its classifier model
        tokenizer  = wrapped.tokenizer
        logger.info("🎯 Using prompt tuning with learnable embeddings")
    else:
        # Standard AutoModel and tokenizer loading from HuggingFace
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, num_labels=3, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # ①  LoRA **every linear layer** inside the encoder
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["dense", "query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    # 1️⃣ Freeze *before* we look at requires_grad flags
    for n, p in model.named_parameters():
        p.requires_grad = (
            "classifier" in n                           # head
            or "lora_" in n                             # adapters
            or "encoder.layer.11." in n                 # last encoder block
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("🔧 Trainable after freezing: %d", trainable)

    # 2️⃣ Build optimiser groups AFTER freezing
    no_decay = ("bias", "LayerNorm.weight")
    optim_groups = [
        { "params": [p for n,p in model.named_parameters()
                     if "prompt_embeddings" in n and p.requires_grad],
          "lr": 1e-3, "weight_decay": 0.0 },
        { "params": [p for n,p in model.named_parameters()
                     if "lora_" in n and p.requires_grad],
          "lr": 5e-4, "weight_decay": 0.0 },
        { "params": [p for n,p in model.named_parameters()
                     if p.requires_grad and
                     not ("prompt_embeddings" in n or "lora_" in n) and
                     not any(nd in n for nd in no_decay)],
          "lr": 2e-5, "weight_decay": 0.01 },
        { "params": [p for n,p in model.named_parameters()
                     if p.requires_grad and
                     not ("prompt_embeddings" in n or "lora_" in n) and
                     any(nd in n for nd in no_decay)],
          "lr": 2e-5, "weight_decay": 0.0 },
    ]
    optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.999), eps=1e-8)

    # Print optimizer group diagnostics
    logger.info("🎯 Custom Optimizer Groups:")
    for i, group in enumerate(optim_groups):
        param_count = sum(p.numel() for p in group["params"])
        lr = group["lr"]
        wd = group["weight_decay"]
        
        # Determine group type
        if i == 0:
            group_type = "Prompt Embeddings"
        elif i == 1:
            group_type = "LoRA Parameters"
        elif i == 2:
            group_type = "Other (with decay)"
        else:
            group_type = "Other (no decay)"
        
        logger.info(f"  Group {i+1} ({group_type}): {param_count:,} params, lr={lr:.0e}, wd={wd}")
    
    total_optim_params = sum(sum(p.numel() for p in group["params"]) for group in optim_groups)
    logger.info(f"🎯 Total optimizer parameters: {total_optim_params:,}")

    # ------------------------------------------------------------------
    # Clamp again (now on the *wrapper's* tokenizer) — this is the one
    # that will actually be used in datasets.map and training.
    # ------------------------------------------------------------------
    SAFE_MAX_LEN = 128            # matches prompt-design in model.py
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 4096:
        logger.warning(
            "Tokenizer model_max_length (%s) is abnormal – resetting to %d",
            tokenizer.model_max_length, SAFE_MAX_LEN,
        )
        tokenizer.model_max_length = SAFE_MAX_LEN

    # ------------------------- device choice -------------------------
    if torch.cuda.is_available():
        run_device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        run_device = torch.device("mps")
    else:
        run_device = torch.device("cpu")
    model.to(run_device)
    # explicit print so you see it even if logging level changes
    print(f"🖥️  Running on device → {run_device}")
    logger.info("🖥️  Using device: %s", run_device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    if args.use_peft:
        peft_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"🔧 LoRA parameters: {peft_params:,} trainable")

    # ──────────────────────────────────────────────
    # Data
    # ──────────────────────────────────────────────
    datasets_dict = load_and_prepare(args.train_file, args.validation_file)
    train_ds = tokenize_dataset(datasets_dict["train"], tokenizer)
    val_ds = tokenize_dataset(datasets_dict["val"], tokenizer)

    # ────────────────────────────────────────────────────────────
    # TrainingArguments – pass only kwargs your local Transformers
    # build understands (older versions miss e.g. evaluation_strategy)
    # ────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # Dynamically choose precision flags (bf16 / fp16) that are both
    # supported by *your GPU* and *this transformers version*.
    # ------------------------------------------------------------------

    want_bf16  = False
    want_fp16  = False
    if run_device.type == "cuda":
        # bf16 only on Ampere+ (A100/H100) or newer; V100 (g4dn) → fp16
        if torch.cuda.is_bf16_supported():
            want_bf16 = True
        else:
            want_fp16 = True
    elif run_device.type == "mps":
        # M-series currently: no mixed-precision in HF trainer → keep FP32
        want_bf16 = want_fp16 = False

    print(f"🧮  Precision  → bf16:{want_bf16}  fp16:{want_fp16}")

    _base_kwargs: dict = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="no",
        save_steps=0,
        logging_steps=10,
        push_to_hub=False,
        disable_tqdm=False,
        remove_unused_columns=False,   # Keep custom columns; avoids Trainer crash
    )
    
    # ③  10% warm-up ➜ cosine decay (so big LRs cool off quickly)
    _base_kwargs.update({
      "warmup_ratio":      0.1,          # 10 %
      "lr_scheduler_type": "cosine",
    })

    # Optional kwargs – add them only if BOTH ① supported by this
    # transformers version *and* ② meaningful for the current device.
    _optional = {
        "evaluation_strategy": "epoch",
        "bf16": want_bf16,
        "fp16": want_fp16,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    for k, v in _optional.items():
        if k in sig.parameters:
            _base_kwargs[k] = v
        else:
            logger.warning("Transformers %s lacks `%s`; skipping it.",
                           transformers.__version__, k)

    targs = TrainingArguments(**_base_kwargs)

    # ------------------------------------------------------------------
    # Build an SFTConfig ***only if we'll use SFTTrainer*** (causal-LMs).
    # For sequence-classification we'll fall back to HF Trainer and this
    # object will stay unused.
    # ------------------------------------------------------------------
    _sft_kwargs = {"packing": args.packing}
    _sig_sft    = inspect.signature(SFTConfig.__init__)
    # optional LoRA knobs
    if "lora_r"     in _sig_sft.parameters:  _sft_kwargs["lora_r"]     = args.lora_r
    if "lora_alpha" in _sig_sft.parameters:  _sft_kwargs["lora_alpha"] = args.lora_alpha
    if "use_peft"   in _sig_sft.parameters:  _sft_kwargs["use_peft"]   = args.use_peft

    # precision flags sometimes live in SFTConfig for ≥0.19
    if "bf16" in _sig_sft.parameters:  _sft_kwargs["bf16"] = want_bf16
    if "fp16" in _sig_sft.parameters:  _sft_kwargs["fp16"] = want_fp16

    sft_cfg = SFTConfig(**_sft_kwargs)

    # ------------------------------------------------------------------
    # Decide which trainer to use
    #   • causal-LM  → keep TRL SFTTrainer
    #   • classifier → plain HF Trainer (avoids the padding bug)
    # ------------------------------------------------------------------

    is_seq_cls = getattr(model.config, "num_labels", 1) > 1

    if is_seq_cls:
        # --- LoRA-wrapped version ---
        logger.info("⚖️  Detected sequence-classification head – wrapping with LoRA adapters")
        from transformers import Trainer, DataCollatorWithPadding

        # 2) data collator
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if run_device.type == "cuda" else None
        )

        # Compute metrics function for per-epoch evaluation
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        # ----- LOSS FUNCTIONS ------------------------------------------------
        class LabelSmoothingLoss(torch.nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing
                self.confidence = 1.0 - smoothing

            def forward(self, logits, target):
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                # per-example negative log-likelihood
                nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
                # per-example uniform smoothing loss
                smooth_loss = -logprobs.mean(dim=-1)
                # combine and average to produce a scalar
                loss = self.confidence * nll_loss + self.smoothing * smooth_loss
                return loss.mean()

        class FocalLoss(torch.nn.Module):
            """
            α : list of per-class weights   (len = n_classes)
            γ : focusing parameter
            """
            def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
                super().__init__()
                self.alpha = torch.tensor(alpha) if alpha is not None else None
                self.gamma = gamma
                self.reduction = reduction

            def forward(self, logits, target):
                ce = torch.nn.functional.cross_entropy(
                        logits, target, reduction="none",
                        weight=self.alpha.to(logits.device) if self.alpha is not None else None)
                pt = torch.exp(-ce)           # prob of true class
                focal = ((1-pt)**self.gamma) * ce
                return focal.mean() if self.reduction=="mean" else focal.sum()

        # -------- CUSTOM TRAINER --------------------------------------------
        class CustomTrainer(Trainer):
            def __init__(self, *args,
                         smoothing=0.05, use_focal=False,
                         focal_alpha=None, focal_gamma=2.0, **kwargs):
                super().__init__(*args, **kwargs)
                self.use_focal = use_focal
                if use_focal:
                    self.loss_fn = FocalLoss(alpha=focal_alpha,
                                             gamma=focal_gamma)
                    logger.info(f"🔹 Using **Focal-Loss** (γ={focal_gamma}, α={focal_alpha})")
                else:
                    self.smoothing = smoothing
                    self.loss_fn = LabelSmoothingLoss(smoothing=smoothing)
                    logger.info(f"🔹 Using **Label-Smoothing** (ε={smoothing})")

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # <-- add **kwargs here
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)
                return (loss, outputs) if return_outputs else loss

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                """Print confusion matrix after each evaluation"""
                super().on_evaluate(args, state, control, metrics, **kwargs)
                if metrics and "eval_predictions" in metrics:
                    preds = metrics["eval_predictions"].argmax(axis=-1)
                    labels = metrics["eval_label_ids"]
                    cm = confusion_matrix(labels, preds)
                    logger.info(f"🔹 Confusion matrix (epoch {state.epoch}):\n{cm}")

            def _save(self, output_dir: str, state_dict=None, **kwargs):
                """Override save to ensure LoRA adapters are saved properly"""
                # Save the LoRA model with adapter config
                self.model.save_pretrained(Path(output_dir))
                # Also save tokenizer only once (root dir)
                if self.tokenizer is not None and not (Path(output_dir)/"tokenizer.json").exists():
                    self.tokenizer.save_pretrained(Path(output_dir))
                logger.info(f"💾 Saved LoRA model to {output_dir}")

        # 3) now use HF Trainer on your LoRA‐wrapped model with custom loss
        trainer = CustomTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,  # Add tokenizer so it gets saved
            optimizers=(optimizer, None),  # Pass custom optimizer (optimizer, lr_scheduler)
            # set `use_focal=True` to activate focal-loss,
            # set `use_focal=False` to revert to label-smoothing
            use_focal=False,
            focal_alpha=[2.0, 2.0, 1.0],   # bearish, bullish, neutral
            focal_gamma=2.0,
            smoothing=0.05,
            compute_metrics=compute_metrics,
        )

    else:
        # ---------- keep the robust SFTTrainer path ----------
        _trainer_kwargs = dict(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=get_peft_config(args) if args.use_peft else None,
            optimizers=(optimizer, None),  # Pass custom optimizer (optimizer, lr_scheduler)
        )

        import trl
        sig_trl = inspect.signature(trl.SFTTrainer.__init__)
        if "tokenizer" in sig_trl.parameters:
            _trainer_kwargs["tokenizer"] = tokenizer
        if "sft_config" in sig_trl.parameters:
            _trainer_kwargs["sft_config"] = sft_cfg
        elif "config" in sig_trl.parameters:
            _trainer_kwargs["config"] = sft_cfg

        trainer = SFTTrainer(**_trainer_kwargs)

    # Confirm optimizer is being used
    logger.info("✅ Custom optimizer passed to trainer")
    logger.info("🚀 Starting fine-tune…")
    trainer.train()                # HF Trainer handles epochs internally
    
    # ────────────────────────────────────────────────────────────────
    # Un-wrap the *real* model (handles PEFT / DataParallel / DDP) once,
    # then put it into eval-mode and save it.
    # ────────────────────────────────────────────────────────────────
    eval_model = trainer.model
    if hasattr(eval_model, "module"):           # DDP / DP wrapper
        eval_model = eval_model.module

    eval_model.eval()
    eval_model = eval_model.to(run_device)

    eval_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("✅ Model saved to %s", args.output_dir)

    # ─── helper: run Val split exactly once & cache results ───────────────
    val_cache: tuple | None = None              # (logits, labels, texts)

    def collect_val_outputs() -> None:
        """Forward-pass the *val* set once and keep logits / labels / text."""
        nonlocal val_cache
        if val_cache is not None:      # already computed
            return                      # → skip
        loader = DataLoader(
            val_ds,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=default_data_collator
        )
        logits, labels, texts = [], [], []
        for batch in tqdm(loader, desc="Val • forward pass"):
            batch_on = {k: v.to(run_device) for k, v in batch.items()}
            with torch.no_grad():
                out = eval_model(**batch_on)
            logits.append(out.logits.cpu())
            labels.extend(batch["labels"].tolist())
            texts.extend(tokenizer.batch_decode(batch["input_ids"],
                                                skip_special_tokens=True))
        logits = torch.cat(logits, dim=0).numpy()
        val_cache = (logits, labels, texts)
        logger.info(f"🔄 Cached validation outputs: {len(labels)} samples")

    # ------------------------------------------------------------------
    # Compute the cache right away so every later consumer is safe.
    # ------------------------------------------------------------------
    collect_val_outputs()

    # ────────────────────────────────────────────────────────────
    # Post-training evaluation on EA (in-sample) and EB (out-of-sample)
    # ────────────────────────────────────────────────────────────
    def evaluate_split(name: str, ds: Dataset):
        loader = DataLoader(
            ds,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=default_data_collator
        )
        preds, labels = [], []
        for batch in tqdm(loader, desc=f"Evaluating {name}"):
            batch = {k: v.to(run_device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = eval_model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        cm = confusion_matrix(labels, preds)
        logger.info(f"🔹 {name} → acc {acc:.3f}  prec {prec:.3f}  rec {rec:.3f}  f1 {f1:.3f}")
        logger.info(f"🔹 {name} confusion matrix:\n{cm}")

    logger.info("🔎 Starting post-training evaluation")
    evaluate_split("EA (in-sample)", train_ds)
    evaluate_split("EB (out-of-sample)", val_ds)

    # ------------------------------------------------------------------
    # ④  Deep-dive error analysis & SHAP on EB split
    # ------------------------------------------------------------------
    logger.info("🔍  Starting confident-error & SHAP analysis …")
    try:
        import shap, itertools, matplotlib   # only loaded if user asked
        from collections import Counter

        # ---------- get logits/preds only ONCE (reuse later) -------------
        collect_val_outputs()  # Ensure val_cache is populated
        
        if val_cache is None:
            raise RuntimeError("val_cache failed to initialize - check collect_val_outputs()")
        
        logits, all_labels, all_text = val_cache          # unpack
        probs  = torch.softmax(torch.tensor(logits), dim=-1)
        conf, preds = probs.max(dim=1)

        wrong_mask = preds != torch.tensor(all_labels)
        df_err = pd.DataFrame({
            "text":  list(itertools.compress(all_text, wrong_mask)),
            "true":  [list(LABEL_MAP.keys())[l] for l in itertools.compress(all_labels, wrong_mask)],
            "pred":  [list(LABEL_MAP.keys())[p] for p in itertools.compress(preds.tolist(), wrong_mask)],
            "conf":  conf[wrong_mask].numpy(),
        }).sort_values("conf", ascending=False)

        top_err = df_err.head(300)
        run_id  = Path(args.output_dir).name
        top_err.to_csv(f"errors_{run_id}.csv", index=False)
        logger.info("💾  errors_%s.csv written (top-300 confident mistakes)", run_id)

        # ---- token histogram for a quick sanity peek
        cc = Counter(w.lower() for t in top_err["text"] for w in t.split())
        logger.info("⚠️  tokens in confident errors: %s",
                    ", ".join(f"{w}({c})" for w, c in cc.most_common(20)))

        # ------------------ SHAP  ---------------------------------------
        masker = shap.maskers.Text(tokenizer)

        def predict_proba(texts):
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=tokenizer.model_max_length
                           ).to(run_device)
            with torch.no_grad():
                out = eval_model(**enc).logits
            return torch.softmax(out, dim=-1).cpu().numpy()

        # Background samples – fall back gracefully if few errors
        bg_text = top_err["text"].sample(30, random_state=0).tolist()
        explainer = shap.Explainer(predict_proba, masker, algorithm="partition",
                                   output_names=list(LABEL_MAP.keys()))
        sv        = explainer(top_err["text"].head(10).tolist())   # 10 examples → quick
        shap.save_html(f"shap_{run_id}.html", sv)
        logger.info("🖼️  SHAP report → shap_%s.html", run_id)

    except Exception as e:
        logger.warning("SHAP / analysis step failed: %s", e)

    # ------------------------------------------------------------------
    # ⑤ Extra visualizations that cost <60s of compute
    # ------------------------------------------------------------------
    logger.info("📊 Generating visualization plots...")
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, auc

        # Reuse cached validation outputs
        collect_val_outputs()  # Ensure val_cache is populated
        
        if val_cache is None:
            raise RuntimeError("val_cache failed to initialize - check collect_val_outputs()")
        
        final_logits, final_labels, _ = val_cache
        final_logits = torch.tensor(final_logits)
        final_preds = torch.argmax(final_logits, dim=-1).tolist()
        final_cm = confusion_matrix(final_labels, final_preds)
        
        # Confusion matrix heatmap
        ConfusionMatrixDisplay(final_cm,
                               display_labels=["Bearish", "Neutral", "Bullish"]
                              ).plot(cmap="Blues")
        plt.title("Final Confusion Matrix")
        plt.savefig(f"cm_final_{Path(args.output_dir).name}.png", dpi=250)
        plt.close()
        
        # Aggregated PR-curve for bullish + bearish (neutral ≈ ignore)
        bear_idx, bull_idx = 0, 2  # bearish=0, bullish=2
        # Create binary labels: 1 if bearish or bullish, 0 if neutral
        y_binary = [1 if label in [bear_idx, bull_idx] else 0 for label in final_labels]
        # Get max probability for bearish or bullish predictions
        probas = final_logits[:, [bear_idx, bull_idx]].max(1).values.numpy()
        
        prec, rec, _ = precision_recall_curve(y_binary, probas)
        pr_auc = auc(rec, prec)
        
        plt.figure(figsize=(8, 6))
        plt.plot(rec, prec, linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR AUC = {pr_auc:.3f} (Bearish + Bullish vs Neutral)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"pr_auc_{Path(args.output_dir).name}.png", dpi=250)
        plt.close()
        
        logger.info(f"📊 Saved visualization plots: cm_final_{Path(args.output_dir).name}.png, pr_auc_{Path(args.output_dir).name}.png")
        
    except Exception as e:
        logger.warning("Visualization step failed: %s", e)


if __name__ == "__main__":
    main()
