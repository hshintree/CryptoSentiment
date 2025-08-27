## CryptoSentiment

A practical, leak‑safe cryptocurrency sentiment research stack centered on BERT‑family models, causal feature engineering, and strict time‑series validation. The codebase implements and extends paper‑style methodology with Triple‑Barrier Labeling (TBL), purged/block time splits, prompt‑based inputs, LoRA fine‑tuning, and ensemble inference.

### Highlights
- **Leak‑free workflow**: per‑fold preprocessing, labeling, and feature generation with temporal gaps; no global fitting on evaluation periods.
- **Time‑series cross‑validation**: date‑wise TimeSeriesSplit with day‑level grouping and purge gaps; blocked/purged CV variants.
- **Features**: RSI/ROC (raw + buckets), causal Previous Label from market regimes, EWMA volatility for TBL.
- **Models**: `ElKulako/cryptobert` baseline; prompt‑based inputs; LoRA adapters and optional prompt embeddings.
- **Training options**: label smoothing, focal loss, AdamW with cosine schedule/warmup, balanced sampling.
- **Evaluation**: per‑fold best‑epoch checkpoints, soft‑probability ensemble, macro metrics, confusion matrices, daily aggregation.
- **SFT**: `sft.py` for supervised fine‑tuning with TRL/HF, SHAP diagnostics, optimizer group control.
- **Reproducibility**: caching, gating of file outputs, deterministic seeds, device selection (CUDA/MPS/CPU).

---

## Repository layout

- `CryptoSentiment_repo/`
  - `trainer.py` — Cross‑validation training loop, per‑fold preprocessing/labeling, checkpointing, ensemble inference.
  - `model.py` — CryptoBERT wrapper, tokenizer/prompt construction, device management, optional LoRA/prompt‑tuning hooks.
  - `preprocessor.py` — Text cleaning, technical indicators (RSI/ROC), raw vs scaled indicators, bucketing, caching.
  - `market_labeler_ewma.py` — EWMA‑based Triple‑Barrier Labeling and `MarketFeatureGenerator` for causal Previous Label.
  - `dataset_loader.py` — Data I/O utilities and optional daily aggregation.
  - `sft.py` — Supervised fine‑tuning entry point (HF/TRL) with LoRA, optimizer control, SHAP/visualizations.
  - `train_one.py` — Single‑run trainer and ensemble helpers for experiments.
  - `tests/` — Runnable scripts for full CV runs, evaluation, quick validations; all now support save‑gating flags.
  - `data/` — Input/output CSVs. Main datasets use stable names: `#1train.csv`, `#2val.csv`, `#3val.csv`.
- `Current/`
  - `dataset_builder.py` — Real‑time dataset builder (Finnhub OHLCV + X/Twitter + Reddit) aligned to 15‑minute candles.
- `requirements.txt` — Core dependencies.
- `config.yaml` — Labeling/model/training configuration.

---

## Core methodology

### Temporal leakage prevention
- All folds are split at the day level. Validation days are separated by a purge gap from training days.
- Within each fold:
  - Fit `Preprocessor` on the training slice; transform validation.
  - Fit `MarketLabelerTBL` on training prices; apply to validation.
  - Generate `Previous Label` causally using `MarketFeatureGenerator` based on completed regimes only.
- No global fitting across EA/EB; evaluation preprocessing and labeling are derived from EA‑fit objects only.

### Datasets
- **EA (train)**: 2020 balanced dataset for model training.
- **EB (eval)**: 2015–2019 and 2021–2023 event‑weighted dataset for out‑of‑sample evaluation.
- Stable filenames produced by generators:
  - EA → `data/#1train.csv` (or `#2train.csv` in the leak‑free generator)
  - EB → `data/#2val.csv` (or `#3val.csv` in the leak‑free generator)

### Features
- RSI(14) and ROC(7/12): raw values for bucketing and scaled values for numeric use.
- Buckets: compact 3‑way (`bearish`, `neutral`, `bullish`). SFT also supports oversold/overbought RSI and graded ROC if needed.
- Previous Label: causal lookup of the last completed regime via barrier end dates.

### Models and training
- Base: `ElKulako/cryptobert` loaded via HF AutoModelForSequenceClassification.
- Prompted inputs: Previous Label, RSI bucket, ROC bucket, and Tweet text concatenated into a single prompt string.
- LoRA: target modules include dense/query/key/value; prompt embeddings optional.
- Losses: label‑smoothing cross‑entropy (default) or focal loss (γ, α per class).
- Optimizer: AdamW with differential parameter groups; cosine schedule with warmup.
- Class imbalance: in‑epoch `WeightedRandomSampler` for balanced batches.

### Evaluation and ensembling
- Save best epoch per fold; store `metrics.json` alongside weights.
- Ensemble by averaging logits across folds, optionally weighted by per‑fold validation F1.
- Produce per‑tweet predictions and daily signals (majority vote or prob‑sum).

---

## Installation

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon: PyTorch MPS kernels are enabled with graceful CPU fallback. CUDA is preferred when available.

---

## Data preparation

Use either generator. Both are leak‑aware and now support save‑gating and optional timestamped filenames.

- Simple generator:
```bash
python CryptoSentiment_repo/generate_eval_datasets.py \
  --no-save           # dry-run; print validation and shapes only
# remove --no-save to write: data/#1train.csv, data/#2val.csv
# add --timestamped to append timestamps to outputs
```

- Leak‑free generator with explicit raw input:
```bash
python CryptoSentiment_repo/generate_eval_datasets_leakfree.py \
  --raw data/combined_dataset_raw.csv \
  --no-save           # dry-run
# remove --no-save to write: data/#2train.csv, data/#3val.csv
# add --timestamped to append timestamps
```



---

## Training and evaluation

### 5‑fold CV training and EB evaluation
```bash
python CryptoSentiment_repo/tests/full_cv_run.py --no-save
# remove --no-save to write signals_ea.csv / signals_eb.csv and daily aggregates
# add --timestamped to suffix filenames
```

### Checkpointed ensemble evaluation
```bash
python CryptoSentiment_repo/tests/evaluate_models.py <timestamp> [sample_n] [csv_path] [--no-save] [--timestamped]
```

### Quick single‑run experiments
```bash
python CryptoSentiment_repo/train_one.py
```

---

## Supervised Fine‑Tuning (SFT)

The `sft.py` script supports LoRA adapters, custom optimizer groups, and SHAP analysis after training. It unwraps PEFT/DP wrappers before saving and uses safe tokenizer max length.

Example:
```bash
python CryptoSentiment_repo/sft.py \
  --model_name_or_path ElKulako/cryptobert \
  --train_file data/#1train.csv \
  --validation_file data/#2val.csv \
  --output_dir checkpoints/cryptobert-bitcoin-sft \
  --learning_rate 2e-5 --num_train_epochs 3 \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
  --use_peft --lora_r 32 --lora_alpha 16
```
Notes:
- Uses label‑smoothing by default; set `use_focal=True` in `CustomTrainer` to switch to focal loss.
- Passes the custom optimizer to HF/TRL Trainer via `optimizers=(optimizer, None)`.
- Produces SHAP HTML and confusion matrix/PR plots after training.

---

## Configuration

Key `config.yaml` sections:
- `market_labeling`: TBL thresholds, vertical window, EWMA parameters, gap days.
- `model`: base checkpoint, prompt/LoRA options, tokenizer limits.
- `training`: epochs, batch size, learning rate, warmup fraction, seeds.

Device detection order: CUDA → MPS → CPU. MPS runs with FP32 and single‑worker DataLoader. CUDA uses pinned memory and multi‑workers.

---

## Outputs

- Checkpoints: `models/ea_<run_id>/foldK/` with `config.json`, weights, tokenizer, `metrics.json`.
- Metrics: `training_metrics.json`, `epoch_metrics.csv`.
- Predictions: `signals_*.csv` and daily aggregates (gated by `--no-save`).
- SFT artifacts: checkpoint directory, `errors_<run_id>.csv`, `cm_final_<run_id>.png`, `pr_auc_<run_id>.png`, `shap_<run_id>.html`.

All save calls in generation and test scripts support `--no-save` and `--timestamped` to limit clutter.

---

## Reproducibility and caching
- Seeds are set across NumPy/PyTorch; repeatability depends on device kernels.
- Preprocessing results for SFT are cached under `cached_data/` to speed up iteration.
- Purged CV and causal feature generation are enforced per fold to avoid temporal bleed.

---

## Troubleshooting
- Extremely high validation accuracy or near‑perfect correlation between `Previous Label` and `Label` suggests leakage; verify per‑fold fitting and purge windows.
- MPS hangs: ensure single DataLoader worker and default float32; the code already applies these.
- Tokenizer errors on long prompts: `sft.py` clamps `model_max_length` to 128 by default.
- Mixed library versions: `sft.py` checks constructor signatures dynamically for HF/TRL compatibility.

---

## Maintenance and organization
To reduce clutter and make discoverability easier:
- Tests and runners live under `CryptoSentiment_repo/tests/` and default to non‑writing unless requested.
- Consider promoting frequently imported modules into a `cryptosentiment/` package with explicit `__init__.py` and type hints.

---

## Citation and background
This work draws on Triple‑Barrier Labeling and sentiment modeling for crypto markets as described in this paper https://arxiv.org/pdf/2502.14897v1. The repository implements a practical, leak‑aware variant with modern HF/TRL tooling. For methodology details, consult the referenced paper and code comments in `trainer.py`, `market_labeler_ewma.py`, and `sft.py`.

If you have any questions feel free to reach out! hshindy@stanford.edu
