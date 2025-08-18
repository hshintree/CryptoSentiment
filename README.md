# CryptoSentiment
Trying to implement this paper for starters: https://arxiv.org/pdf/2502.14897v1. Will try and innovate after.
Main Pipeline

main.py orchestrates the end‑to‑end flow:

Load configuration.

Read raw event and tweet datasets.

Run preprocessing (text cleanup plus optional technical indicators).

Apply market labeling.

Initialize a BERT-based model and train it using PyTorch.

Evaluate the trained model.

Generate trading signals and run a backtest.

These stages are executed sequentially with logging, as shown in the code from lines 17 onward.

Outputs (preprocessed files and signals) are saved to the location specified in the config before the program exits.

Key Components

DatasetLoader – Reads PreBit CSVs using paths from the config. It supports the directory layout with yearly tweet files and price labels or a single combined CSV, and can aggregate tweets per day.

Preprocessor – Applies basic NLP cleaning steps (lowercasing, stripping URLs/usernames, removing punctuation, optional lemmatization) and computes technical indicators (RSI, ROC) if price columns exist. Missing values are forward/backward filled, and indicators are min–max normalized. 

MarketLabeler – Implements the Triple Barrier Labeling (TBL) approach for short‑term trend detection. Volatility is estimated with an exponential moving standard deviation and is used to scale profit‑taking (upper) and stop‑loss (lower) price barriers. A vertical barrier enforces an observation window of 8‑15 days. Each tweet’s market trend is labeled according to the first barrier touched within that window, yielding “Bullish”, “Bearish”, or “Neutral”.

Model – Wraps either CryptoBERT or FinBERT via HuggingFace. `preprocess_input` embeds market context (date, previous label, RSI, ROC) directly into a text prompt before tokenization.

Trainer – Performs group 5‑fold cross‑validation using PyTorch. The first 11 layers of CryptoBERT are frozen and only the classification head is updated. Training uses the AdamW optimizer with a warmup schedule.

Evaluation – Computes accuracy, precision, recall and F1 using scikit‑learn after generating predictions with the model. 

SignalGenerator – Turns per‑tweet predictions into daily trading signals using aggregation methods such as majority vote or mean, then optionally merges them into a combined signal. 

Backtester – Uses vectorbt to simulate trading strategies on different predefined market regimes. It outputs metrics like Sharpe ratio, Sortino ratio, drawdown and total return. 

Pointers for Next Steps

Data Acquisition & Paths – The configuration now supports the PreBit multimodal dataset. You can either provide `prebit_dataset_path` pointing to a single combined CSV or set `prebit_dataset_dir` to the folder containing the yearly `combined_tweets_*_labeled.csv` files and `price_label.csv`. When either PreBit option is supplied the separate tweet and event files are ignored.

Deep Learning Environment – The pipeline has migrated to PyTorch. Install PyTorch and the HuggingFace transformers library. A `requirements.txt` file is recommended.

Understanding Triple Barrier Labeling – The MarketLabeler provides only a basic implementation. Learning the underlying triple-barrier method from the referenced paper will help refine the labeling strategy and tune parameters such as barrier distances or volatility estimation.

VectorBT Backtesting – The backtester uses stubbed market regimes and dummy price data. Explore vectorbt to understand how to incorporate real historical prices and more sophisticated strategies.

Expand Evaluation and Experimentation – The model currently outputs BERT hidden states without a classification head. Adding a small neural head or fine-tuning the BERT weights will likely be required for real predictive power. Also consider adding experiments, hyperparameter tuning, and cross-validation for the signal generation thresholds.

Project Organization – Tests and a comprehensive README are missing. Adding unit tests for each module and extending the documentation would make the codebase easier to maintain.

## Setup
Use Python 3.10 in a new conda environment:
```bash
conda create -n cryptosentiment python=3.10
conda activate cryptosentiment
pip install -r requirements.txt
```

By following these pointers—configuring real data, installing the necessary libraries, and gradually refining each stage—you can evolve the current skeleton into a fully functional crypto‑sentiment trading pipeline.

## Running Tests

Basic sanity tests are provided using `pytest`. After installing the project dependencies you can run:

```bash
python -m py_compile CryptoSentiment_repo/*.py
pytest -q
```

This compiles the source files and executes the unit tests found in the `tests/` directory.
