## trainer.py

import tensorflow as tf
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from typing import Any, Dict
from model import Model  # Assuming Model class from model.py
from market_labeler import MarketLabeler  # For accessing labeled data

class Trainer:
    """Trainer class for handling the training process of BERT-based models."""

    def __init__(self, model: Model, data: pd.DataFrame, config_path: str = 'config.yaml'):
        """
        Initialize the Trainer with a model, a labeled dataset, and configurations.

        Args:
            model (Model): Instance of the model to be trained.
            data (pd.DataFrame): Labeled dataset using market behaviors.
            config_path (str): Path to the configuration YAML file.
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Recording model and data
        self.model = model
        self.data = data

        # Extract training configurations
        training_config = self.config['training']
        self.learning_rate = training_config.get('learning_rate', 1e-5)
        self.batch_size = training_config.get('batch_size', 12)
        self.epochs = training_config.get('epochs', 2)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.warmup_steps = self.config['training'].get('warmup_steps', 0.1)

    def train(self) -> None:
        """Execute training over the defined number of epochs."""
        data = self._prepare_data(self.data)
        gkf = GroupKFold(n_splits=5)  # Use Group 5-fold cross-validation

        for fold, (train_idx, val_idx) in enumerate(gkf.split(data, groups=data['group'])):
            print(f"Fold {fold + 1}/{gkf.n_splits}")

            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]

            # Prepare TensorFlow datasets
            train_dataset = self._create_tf_dataset(train_data)
            val_dataset = self._create_tf_dataset(val_data)

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")
                
                # Training Loop
                for step, (inputs, labels) in enumerate(train_dataset):
                    with tf.GradientTape() as tape:
                        outputs = self.model.forward(inputs)
                        loss = self._compute_loss(outputs, labels)

                    gradients = tape.gradient(loss, self.model.bert_model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.bert_model.trainable_variables))

                    if step % 10 == 0:
                        print(f"Step {step}, Loss: {loss.numpy()}")

                # Validation
                self._evaluate(val_dataset)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training including label labeling."""
        labeler = MarketLabeler()
        labeled_data = labeler.label_data(data)
        labeled_data['group'] = self._assign_groups(labeled_data)
        return labeled_data

    def _assign_groups(self, data: pd.DataFrame) -> np.ndarray:
        """Assign groups for cross-validation to prevent leakage."""
        return shuffle(data.index.to_numpy()) // (len(data) // 5)

    def _create_tf_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """Convert DataFrame into a TensorFlow dataset suitable for training."""
        features = data[['Tweet Content', 'RSI', 'ROC', 'Tweet Date', 'Previous Label']].to_dict('records')
        labels = data['Label'].apply(lambda x: 0 if x == 'Bearish' else (1 if x == 'Neutral' else 2)).values
        
        # Encode features and labels as a tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        return dataset

    def _compute_loss(self, logits: tf.Tensor, labels: np.ndarray) -> tf.Tensor:
        """Compute the loss using SparseCategoricalCrossentropy."""
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss_fn(labels, logits)

    def _evaluate(self, dataset: tf.data.Dataset) -> None:
        """Evaluate the model on validation dataset."""
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        for inputs, labels in dataset:
            predictions = self.model.forward(inputs)
            accuracy_metric.update_state(labels, predictions)

        print(f"Validation Accuracy: {accuracy_metric.result().numpy()}")
