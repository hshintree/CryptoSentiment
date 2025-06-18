"""model.py – dynamic max_length fix
• Now reads config.max_position_embeddings at runtime
• Caps token length to the *minimum* of 256 and max_position_embeddings
"""

import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Model:
    """BERT-based model wrapper for financial sentiment analysis."""

    def __init__(self, params: dict | str):
        if isinstance(params, str):
            import pathlib
            with open(pathlib.Path(params)) as f:
                params = yaml.safe_load(f)["model"]

        self.model_type       = params.get("type", "CryptoBERT")
        self._n_freeze_layers = int(params.get("freeze_layers", 11))

        if self.model_type == "CryptoBERT":
            ckpt = "vinai/bertweet-base"
        elif self.model_type == "FinBERT":
            ckpt = "yiyanghkust/finbert-tone"
        else:
            raise ValueError("Unsupported model type – choose CryptoBERT or FinBERT")

        self.tokenizer  = AutoTokenizer.from_pretrained(ckpt)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            ckpt, num_labels=3
        )

    def preprocess_input(
        self,
        *,
        tweet_content: str,
        rsi: float,
        roc: float,
        date: str,
        previous_label: str,
    ) -> dict:
        """
        Tokenise prompt and return fixed-length tensors.

        Dynamically queries the model's max_position_embeddings so we
        never exceed the embedding-matrix range, even if it's <256.
        """
        prompt = (
            f"Date:{date} | Prev:{previous_label} | "
            f"ROC:{roc:.4f} | RSI:{rsi:.2f} | Tweet:{tweet_content}"
        )
        # determine safe length
        max_pos = self.bert_model.config.max_position_embeddings
        max_len = min(128, max_pos)  # Reduced from 256 to 128 to be safe

        # Get tokenizer output
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        
        # Create position IDs
        position_ids = torch.arange(max_len, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand_as(tokenized["input_ids"])
        tokenized["position_ids"] = position_ids
        
        return tokenized

    def forward(self, inputs):
        """Forward pass returning logits."""
        return self.bert_model(**inputs).logits

    def freeze_layers(self, freeze_until: int | None = None) -> None:
        """Freeze lower transformer blocks."""
        if freeze_until is None:
            freeze_until = self._n_freeze_layers
        encoder = self.bert_model.base_model.encoder
        for layer in encoder.layer[:freeze_until]:
            for p in layer.parameters():
                p.requires_grad = False
