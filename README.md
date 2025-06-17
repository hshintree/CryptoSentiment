# CryptoSentiment
Trying to implement this paper for starters: https://arxiv.org/pdf/2502.14897v1. Will try and innovate after.
Main Pipeline

main.py orchestrates the end‑to‑end flow:

Load configuration.

Read raw event and tweet datasets.

Run preprocessing (text cleanup plus optional technical indicators).

Apply market labeling.

Initialize a BERT-based model and train it.

Evaluate the trained model.

Generate trading signals and run a backtest.

These stages are executed sequentially with logging, as shown in the code from lines 17 onward.

Outputs (preprocessed files and signals) are saved to the location specified in the config before the program exits.

Key Components

DatasetLoader – Reads CSV files from the paths in the config, checking that required columns exist. 

Preprocessor – Applies basic NLP cleaning steps (lowercasing, stripping URLs/usernames, removing punctuation, optional lemmatization) and computes technical indicators (RSI, ROC) if price columns exist. Missing values are forward/backward filled, and indicators are min–max normalized. 

MarketLabeler – Implements the “Triple Barrier Labeling” approach. It estimates volatility from log returns and sets dynamic price barriers, then assigns a label (“Bullish”, “Bearish”, “Neutral”) based on which barrier gets hit first within a fixed time window. 

Model – Wraps either CryptoBERT or FinBERT via HuggingFace. preprocess_input embeds market context (date, previous label, RSI, ROC) directly into a text prompt before tokenization. The forward pass simply runs the BERT model and returns the hidden states. 

Trainer – Performs group 5‑fold cross‑validation using TensorFlow and calculates loss via SparseCategoricalCrossentropy. Training batches are produced from the labeled DataFrame. 

Evaluation – Computes accuracy, precision, recall and F1 using scikit‑learn after generating predictions with the model. 

SignalGenerator – Turns per‑tweet predictions into daily trading signals using aggregation methods such as majority vote or mean, then optionally merges them into a combined signal. 

Backtester – Uses vectorbt to simulate trading strategies on different predefined market regimes. It outputs metrics like Sharpe ratio, Sortino ratio, drawdown and total return. 

Pointers for Next Steps

Data Acquisition & Paths – The config file currently contains placeholder paths (e.g., path/to/bitcoin_historical_events.csv). Gathering real data, placing it in those paths, and adjusting the config accordingly is a prerequisite for running the pipeline.

Deep Learning Environment – The project uses TensorFlow and HuggingFace models; setting up an environment with the required versions of these libraries is necessary. The repository has no dependency list yet, so creating a requirements.txt or similar would help.

Understanding Triple Barrier Labeling – The MarketLabeler provides only a basic implementation. Learning the underlying triple-barrier method from the referenced paper will help refine the labeling strategy and tune parameters such as barrier distances or volatility estimation.

VectorBT Backtesting – The backtester uses stubbed market regimes and dummy price data. Explore vectorbt to understand how to incorporate real historical prices and more sophisticated strategies.

Expand Evaluation and Experimentation – The model currently outputs BERT hidden states without a classification head. Adding a small neural head or fine-tuning the BERT weights will likely be required for real predictive power. Also consider adding experiments, hyperparameter tuning, and cross-validation for the signal generation thresholds.

Project Organization – Tests and a comprehensive README are missing. Adding unit tests for each module and extending the documentation would make the codebase easier to maintain.

By following these pointers—configuring real data, installing the necessary libraries, and gradually refining each stage—you can evolve the current skeleton into a fully functional crypto‑sentiment trading pipeline.
