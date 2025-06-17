## trainer.py

import torch
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
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
        self.optimizer = AdamW(self.model.bert_model.parameters(), lr=self.learning_rate)
        self.warmup_steps = self.config['training'].get('warmup_steps', 0.1)
        self.scheduler = None

    def train(self) -> None:
        """Execute training over the defined number of epochs."""
        data = self._prepare_data(self.data)
        gkf = GroupKFold(n_splits=5)  # Use Group 5-fold cross-validation

        for fold, (train_idx, val_idx) in enumerate(gkf.split(data, groups=data['group'])):
            print(f"Fold {fold + 1}/{gkf.n_splits}")

            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]

            # Prepare PyTorch datasets
            train_loader = self._create_torch_loader(train_data, shuffle=True)
            val_loader = self._create_torch_loader(val_data)

            total_steps = len(train_loader) * self.epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(total_steps * self.warmup_steps),
                num_training_steps=total_steps,
            )

            self.model.freeze_layers(11)
            self.model.bert_model.train()

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")

                # Training Loop
                for step, batch in enumerate(train_loader):
                    labels = batch.pop('labels')
                    outputs = self.model.bert_model(**batch)
                    loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if step % 10 == 0:
                        print(f"Step {step}, Loss: {loss.item()}")

                # Validation
                self._evaluate(val_loader)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training including label labeling."""
        labeler = MarketLabeler()
        labeled_data = labeler.label_data(data)
        labeled_data['group'] = self._assign_groups(labeled_data)
        return labeled_data

    def _assign_groups(self, data: pd.DataFrame) -> np.ndarray:
        """Assign groups for cross-validation to prevent leakage."""
        return shuffle(data.index.to_numpy()) // (len(data) // 5)

    def _create_torch_loader(self, data: pd.DataFrame, shuffle: bool = False) -> DataLoader:
        """Convert DataFrame into a PyTorch DataLoader."""
        encodings = []
        labels = []
        for _, row in data.iterrows():
            enc = self.model.preprocess_input(
                tweet_content=row['Tweet Content'],
                rsi=row['RSI'],
                roc=row['ROC'],
                date=row['Tweet Date'],
                previous_label=row['Previous Label'],
            )
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            encodings.append(enc)
            label = 0 if row['Label'] == 'Bearish' else (1 if row['Label'] == 'Neutral' else 2)
            labels.append(label)

        class DS(Dataset):
            def __init__(self, enc, lab):
                self.enc = enc
                self.lab = lab

            def __len__(self):
                return len(self.lab)

            def __getitem__(self, idx):
                item = {k: v for k, v in self.enc[idx].items()}
                item['labels'] = torch.tensor(self.lab[idx])
                return item

        dataset = DS(encodings, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _evaluate(self, loader: DataLoader) -> None:
        """Evaluate the model on validation dataset."""
        self.model.bert_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop('labels')
                outputs = self.model.bert_model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total else 0
        print(f"Validation Accuracy: {accuracy:.4f}")
        self.model.bert_model.train()

