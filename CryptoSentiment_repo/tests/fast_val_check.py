#!/usr/bin/env python3
"""
Fast validation / stress-test on the 2021 VAL set with lots of
post-hoc tricks (temp-scaling, priors, full-prob aggregation, tiny
LoRA refresh).

Usage examples
--------------
# plain run on 25 k stratified tweets
python fast_val_check.py 20250621_074712 --sample 25000

# temperature grid & Bayesian priors
python fast_val_check.py 20250621_074712 --temp search --bayes

# keep per-tweet probs + smarter daily vote
python fast_val_check.py 20250621_074712 --save-probs

# tiny one-epoch LoRA refresh on a week of Jan-2021 tweets
python fast_val_check.py 20250621_074712 --lora val_week_jan21.csv
"""
import argparse, sys, numpy as np, pandas as pd, torch, math, json
from glob       import glob
from pathlib    import Path
from tqdm.auto  import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BitsAndBytesConfig
)

try:
    from peft import LoraConfig, get_peft_model   # only if --lora
except ImportError:
    LoraConfig = get_peft_model = None            # will warn later

# ------------------------------------------------------------------ CLI
ap = argparse.ArgumentParser()
ap.add_argument("timestamp",
    help="folder timestamp, e.g. 20250621_074712 (under models/ea_2019-2020_<ts>)")
ap.add_argument("--sample", type=int, default=25000,
    help="how many VAL tweets to keep (stratified; 0 = all) [25 000]")
ap.add_argument("--temp",   default="1.0",
    help="'search' to grid-search 0.7-1.3, or a float (e.g. 0.85)")
ap.add_argument("--bayes",  action="store_true",
    help="apply class-prior re-weighting")
ap.add_argument("--save-probs", action="store_true",
    help="save p_bear/p_neut/p_bull columns")
ap.add_argument("--lora",
    help="CSV with a small 2021 sub-set for LoRA refresher (optional)")
args = ap.parse_args()

BASE = Path("models") / f"ea_2019-2020_{args.timestamp}"
val_path = sorted(glob("data/val_dataset_2021_stress_*.csv"))[-1]
VAL = (pd.read_csv(val_path)
         .rename(columns={"date": "Tweet Date"}))
VAL["Tweet Date"] = pd.to_datetime(VAL["Tweet Date"], errors="coerce")

# -------- cheap stratified sampling -----------------------------------
if args.sample and args.sample < len(VAL):
    per_month = math.ceil(args.sample / 12)
    sampled = []
    for m, g in VAL.groupby(VAL["Tweet Date"].dt.month):
        want = min(per_month, len(g))
        sampled.append(g.sample(want, random_state=42))
    VAL = pd.concat(sampled).sort_values("Tweet Date").reset_index(drop=True)
    print(f"🔹 Stratified sample: {len(VAL):,} tweets "
          f"({per_month} / month max)")
else:
    print(f"🔹 Using full VAL: {len(VAL):,} tweets")

# --------------------- device & precision tweaks ----------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"🔹 Running inference on {DEVICE}")

torch.set_default_dtype(torch.float16 if DEVICE.type != "cpu" else torch.float32)
# channels-last speeds up MPS/AMP a bit
memory_fmt = torch.channels_last if DEVICE.type != "cpu" else torch.contiguous_format

# ------------------ helper ------------------------------------------------
def softmax_(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

classes = np.array(["Bearish", "Neutral", "Bullish"])
probs_accum = np.zeros((len(VAL), 3), dtype=np.float32)

# ------------------ LoRA tiny refresh data (optional) -----------------
def lora_refresh(model, tokenizer, csv_path):
    if LoraConfig is None:
        print("⚠️  peft not installed – skipping LoRA")
        return model
    small = pd.read_csv(csv_path).sample(frac=1.0, random_state=1)
    texts = small["Tweet Content"].tolist()
    labels= small["Label"].map({"Bearish":0,"Neutral":1,"Bullish":2}).tolist()

    cfg = LoraConfig(r=8, lora_alpha=16, bias="none",
                     target_modules=["classifier"])
    model = get_peft_model(model, cfg)
    model.train().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    bs  = 32
    for epoch in range(1):
        for i in range(0, len(texts), bs):
            chunk_t = texts[i:i+bs]
            ys      = torch.tensor(labels[i:i+bs], device=DEVICE)
            enc     = tokenizer(chunk_t, truncation=True, padding=True,
                                max_length=128, return_tensors="pt").to(DEVICE)
            out     = model(**enc)
            loss    = torch.nn.functional.cross_entropy(out.logits, ys)
            loss.backward(); opt.step(); opt.zero_grad()
    model.eval()
    return model

# ------------------ iterate over fold checkpoints --------------------
for fold_dir in sorted(BASE.glob("fold[0-9]*")):
    if not any((fold_dir / n).exists() for n in
               ("pytorch_model.bin","model.safetensors")):
        print(f"⚠️  {fold_dir.name} missing weights – skipping"); continue

    print(f"🔗 Loading {fold_dir.name} …")
    tokenizer = AutoTokenizer.from_pretrained(fold_dir)

    # use 4-bit if on CPU for speed / mem
    quant_cfg = (BitsAndBytesConfig(load_in_4bit=True)
                 if DEVICE.type == "cpu" else None)

    model = AutoModelForSequenceClassification.from_pretrained(
                fold_dir,
                torch_dtype=torch.float16 if DEVICE.type != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
                device_map={"":DEVICE} if quant_cfg is None else None,
                quantization_config=quant_cfg,
            ).to(DEVICE, memory_format=memory_fmt).eval()

    # optional tiny LoRA touch-up
    if args.lora:
        model = lora_refresh(model, tokenizer, args.lora)

    # ---------- batched inference ---------------
    bs = 256 if DEVICE.type != "cpu" else 64
    for i in tqdm(range(0, len(VAL), bs), desc=fold_dir.name):
        chunk = VAL.iloc[i:i+bs]
        enc   = tokenizer(chunk["Tweet Content"].tolist(),
                          truncation=True, padding=True,
                          max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits.float().cpu().numpy()
        probs_accum[i:i+bs] += softmax_(logits)

# ------------------ step 1  temperature scaling ----------------------
temps = np.arange(0.7,1.31,0.05) if args.temp=="search" else [float(args.temp)]
best_acc, best_T = -1, 1.0
for T in temps:
    scaled = probs_accum / T
    pred   = classes[scaled.argmax(-1)]
    acc    = (pred == VAL["Label"]).mean() if "Label" in VAL.columns else np.nan
    if acc>best_acc: best_acc,best_T=acc,T
    if args.temp=="search":
        print(f"T={T:.2f}  acc={acc:.3f}")
print(f"🔸 Using T={best_T:.2f}")
probs = probs_accum / best_T

# ------------------ step 2  Bayesian prior bump ----------------------
if args.bayes and "Label" in VAL.columns:
    pri = VAL["Label"].value_counts(normalize=True).reindex(classes).fillna(0.01)
    logp = np.log(pri.values)
    probs = softmax_(np.log(probs+1e-9) + logp)
    print("🔸 Applied class priors")

# ------------------ step 3  per-tweet & daily signals ---------------
VAL[["p_bear","p_neut","p_bull"]] = probs / probs.sum(1,keepdims=True)

VAL["Pred_Label"] = classes[probs.argmax(-1)]
VAL["Pred_Conf"]  = probs.max(-1)

# optionally drop prob columns
if not args.save_probs:
    VAL = VAL.drop(columns=["p_bear","p_neut","p_bull"])

VAL.to_csv("signals_val_2021.csv", index=False)
print("✅  signals_val_2021.csv written")

# smarter daily vote = highest *sum* of probs
daily = (
    VAL.assign(day=VAL["Tweet Date"].dt.normalize())
       .groupby("day")[["p_bear","p_neut","p_bull"]]
       .sum()
       .idxmax(axis=1)
       .to_frame("Daily_Signal")
)
daily.to_csv("val_daily_signals.csv")
print("✅  val_daily_signals.csv written")

# quick accuracy
if "Label" in VAL.columns:
    acc = (VAL["Pred_Label"] == VAL["Label"]).mean()
    print(f"⚡  raw VAL accuracy : {acc:.3f}")

print("✨ done")
