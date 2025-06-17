## model.py

import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import yaml

class Model:
    """Model class for creating and managing a BERT-based language model for financial sentiment analysis."""

    def __init__(self, params: dict):
        """
        Initialize the BERT model for financial sentiment analysis.
        
        Args:
            params (dict): Contains configurations such as model type and prompt tuning settings.
        """
        # Load configuration file for model settings
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Retrieve settings from configuration
        self.model_type = params.get('type', 'CryptoBERT')
        self.prompt_tuning = params.get('prompt_tuning', True)
        
        # Initialize the tokenizer and the model based on the type specified
        if self.model_type == 'CryptoBERT':
            self.tokenizer = BertTokenizer.from_pretrained('vinai/bertweet-base')  # Example model, can be replaced
            self.bert_model = TFBertModel.from_pretrained('vinai/bertweet-base')
        elif self.model_type == 'FinBERT':
            self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')  # Example model, can be replaced
            self.bert_model = TFBertModel.from_pretrained('yiyanghkust/finbert-tone')
        else:
            raise ValueError("Unsupported model type. Choose 'CryptoBERT' or 'FinBERT'.")

    def preprocess_input(self, tweet_content: str, rsi: float, roc: float, date: str, previous_label: str) -> dict:
        """
        Preprocess input data by embedding market context into text.
        
        Args:
            tweet_content (str): The raw tweet content.
            rsi (float): Relative Strength Index value.
            roc (float): Rate of Change value.
            date (str): Date as a string.
            previous_label (str): Label from previous prediction.

        Returns:
            dict: Encoded inputs suitable for model.
        """
        # Create a prompt with market-related context
        prompt = f"Date: {date}, Previous Label: {previous_label}, ROC: {roc}, RSI: {rsi}, Tweet: {tweet_content}"
        
        # Encode the text with tokenizer
        inputs = self.tokenizer(prompt, return_tensors="tf", padding=True, truncation=True)
        
        return inputs

    def forward(self, inputs):
        """
        Perform a forward pass through the model.
        
        Args:
            inputs (dict): Tokenized inputs as tensors.

        Returns:
            tf.Tensor: Output logits from the model.
        """
        # Obtain model output
        outputs = self.bert_model(inputs)
        return outputs.last_hidden_state
