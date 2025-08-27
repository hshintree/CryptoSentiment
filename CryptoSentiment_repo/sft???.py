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
from gpu_scripts.preprocessor import Preprocessor
from gpu_scripts.market_labeler_ewma import MarketLabelerTBL, MarketFeatureGenerator
from gpu_scripts.model import Model
from gpu_scripts.train_one import _coerce_dates

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LABEL_MAP = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
RSI_BUCKETS = {"oversold": (None, 30), "neutral": (30, 70), "overbought": (70, None)}


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
    if val >= thr + 2 * sigma:
        return "rising fast"
    if thr + sigma <= val < thr + 2 * sigma:
        return "rising"
    if thr - 2 * sigma <= val <= thr - sigma:
        return "falling"
    if val < thr - 2 * sigma:
        return "falling fast"
    return "neutral"


def build_prompt(row: pd.Series) -> str:
    """Construct the pure‑text prompt (no date)."""
    return (
        f"Previous Label: {row['Previous Label']}, "
        f"RSI condition: {row['RSI_bucket']}, Price momentum: {row['ROC_bucket']}, "
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

    # Quantisation
    parser.add_argument("--load_in_4bit", action="store_true")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # ensure output directory exists BEFORE trainer tries to write there
    # ------------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer & model
    logger.info("🔧 Loading base model and tokenizer for LoRA …")

    # Standard AutoModel and tokenizer loading from HuggingFace
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=3, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Apply LoRA correctly to this model
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["dense", "out_proj", "query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    # Print trainable parameter count after LoRA wrap
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("🔧 Trainable params after LoRA wrap: %d", trainable)

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

    # Data
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
        save_strategy="epoch",
        logging_steps=10,
        push_to_hub=False,
        disable_tqdm=False,
    )
    
    # 1) warm up for 20% of total steps, 2) cosine-decay to 0
    steps_per_epoch = len(train_ds) // (
       args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps)
    )
    total_steps     = steps_per_epoch * args.num_train_epochs
    _base_kwargs.update({
      "warmup_steps":       int(total_steps * 0.2),
      "lr_scheduler_type":  "cosine",
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

        # Custom Label smoothing cross-entropy loss function
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

        # Compute metrics function for per-epoch evaluation
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        class LabelSmoothingTrainer(Trainer):
            def __init__(self, *args, smoothing=0.05, **kwargs):
                super().__init__(*args, **kwargs)
                self.smoothing = smoothing
                self.loss_fn = LabelSmoothingLoss(smoothing=self.smoothing)

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
                if hasattr(self.model, 'save_pretrained'):
                    self.model.save_pretrained(output_dir)
                # Also save tokenizer
                if hasattr(self, 'tokenizer'):
                    self.tokenizer.save_pretrained(output_dir)
                logger.info(f"💾 Saved LoRA model to {output_dir}")

        # 3) now use HF Trainer on your LoRA‐wrapped model with custom loss
        trainer = LabelSmoothingTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,  # Add tokenizer so it gets saved
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

    logger.info("🚀 Starting fine-tune…")
    trainer.train()                # HF Trainer handles epochs internally
    trainer.save_model(args.output_dir)
    logger.info("✅ Model saved to %s", args.output_dir)

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
                outputs = model(**batch)
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


if __name__ == "__main__":
    main()
