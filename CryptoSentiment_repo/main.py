## main.py

import logging
import os
import yaml
import pandas as pd

from CryptoSentiment_repo.dataset_old_loader import DatasetLoader
from preprocessor import Preprocessor
from market_labeler import MarketLabeler
from model import Model
from trainer import Trainer
from evaluation import Evaluation
from signal_generator import SignalGenerator
from backtester import Backtester

def main(config_path: str = 'config.yaml'):
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Step 1: Data Loading
        logging.info("Loading data...")
        data_loader = DatasetLoader(config_path=config_path)
        dataset = data_loader.load_dataset()
        event_data = dataset
        tweet_data = dataset

        # Step 2: Data Preprocessing
        logging.info("Preprocessing data...")
        preprocessor = Preprocessor(config_path=config_path)
        preprocessed_events = preprocessor.preprocess(event_data)
        preprocessed_tweets = preprocessor.preprocess(tweet_data)

        # Step 3: Market-Derived Labeling
        logging.info("Applying market-derived labeling...")
        market_labeler = MarketLabeler(config_path=config_path)
        labeled_events = market_labeler.label_data(preprocessed_events)

        # Step 4: Model Initialization and Training
        logging.info("Initializing and training model...")
        model_params = config.get('model', {})
        model = Model(params=model_params)
        trainer = Trainer(model=model, data=labeled_events, config_path=config_path)
        trainer.train()

        # Step 5: Model Evaluation
        logging.info("Evaluating model...")
        evaluator = Evaluation(model=model, data=labeled_events, config_path=config_path)
        evaluation_results = evaluator.evaluate()

        # Step 6: Signal Generation
        logging.info("Generating trading signals...")
        signal_generator = SignalGenerator(model=model, config_path=config_path)
        trading_signals = signal_generator.generate_signals(preprocessed_tweets)

        # Step 7: Backtesting
        logging.info("Running backtests...")
        backtester = Backtester(signals=trading_signals, config_path=config_path)
        backtest_results = backtester.run_backtest()

        # Logging results
        logging.info("Evaluation Results:")
        for metric, value in evaluation_results.items():
            logging.info(f"{metric}: {value}")

        logging.info("Backtesting Results:")
        for regime, metrics in backtest_results.items():
            logging.info(f"Market Regime {regime}")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value}")

        # Step 8: Save outputs if required
        output_data_path = config['data'].get('output_data_path', 'path/to/output')
        if not os.path.exists(output_data_path):
            os.makedirs(output_data_path)

        preprocessed_events.to_csv(os.path.join(output_data_path, 'preprocessed_events.csv'), index=False)
        preprocessed_tweets.to_csv(os.path.join(output_data_path, 'preprocessed_tweets.csv'), index=False)
        trading_signals.to_csv(os.path.join(output_data_path, 'trading_signals.csv'), index=False)

        # Log completion
        logging.info("Pipeline execution completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
