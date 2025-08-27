## evaluation.py

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any
from gpu_scripts.model import Model  # Assuming Model class from model.py

class Evaluation:
    """Class to evaluate the performance of a trained language model on market-derived labeled data."""
    
    def __init__(self, model: Model, data: pd.DataFrame, config_path: str = 'config.yaml'):
        """
        Initialize the Evaluation class with a trained model and the dataset for evaluation.
        
        Args:
            model (Model): The trained language model to be evaluated.
            data (pd.DataFrame): Data on which the model's performance will be evaluated.
            config_path (str): Path to the configuration YAML file.
        """
        # Load configuration for evaluation settings
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = model
        self.data = data
        self.evaluation_metrics = self.config['evaluation'].get('metrics', ['accuracy', 'precision', 'recall', 'f1_score'])

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance using specified metrics.
        
        Returns:
            dict: A dictionary containing evaluation metrics and their values.
        """
        predictions = self._generate_predictions()
        true_labels = self.data['Label'].apply(lambda x: 0 if x == 'Bearish' else (1 if x == 'Neutral' else 2)).values
        
        results = {}
        
        # Calculate each metric specified in the config
        if 'accuracy' in self.evaluation_metrics:
            results['accuracy'] = accuracy_score(true_labels, predictions)
        if 'precision' in self.evaluation_metrics:
            results['precision'] = precision_score(true_labels, predictions, average='macro')
        if 'recall' in self.evaluation_metrics:
            results['recall'] = recall_score(true_labels, predictions, average='macro')
        if 'f1_score' in self.evaluation_metrics:
            results['f1_score'] = f1_score(true_labels, predictions, average='macro')
        
        # Logging
        for metric, value in results.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return results

    def _generate_predictions(self) -> np.ndarray:
        """
        Forward the dataset through the model to obtain predictions.
        
        Returns:
            np.ndarray: Array of predicted labels.
        """
        predictions = []
        for i, row in self.data.iterrows():
            rsi_bucket = row.get("RSI_bucket", "neutral")
            roc_bucket = row.get("ROC_bucket", "neutral")
            inputs = self.model.preprocess_input(
                tweet_content=row['Tweet Content'],
                previous_label=row['Previous Label'],
                rsi_bucket=rsi_bucket,
                roc_bucket=roc_bucket,
            )
            logits = self.model.forward(inputs)
            predicted_label = logits.argmax(dim=-1).item()
            predictions.append(predicted_label)
        
        return np.array(predictions)

