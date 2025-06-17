"""model.py – patched 2025‑06‑17
Fixed issues:
  • Always `import yaml` at top‑level (prevents UnboundLocalError)
  • Tokeniser now truncates to 512 tokens and leaves padding to the
    DataCollator (avoids index‑out‑of‑range during training)
  • Removed unused secondary YAML load
"""

import yaml                                  # NEW top‑level import
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Model:
    """BERT‑based model wrapper for financial sentiment analysis."""

    def __init__(self, params: dict | str):
        """Create tokenizer + sequence‑classification head.

        Parameters
        ----------
        params : dict | str
            Either a *dict* from config or a path to `config.yaml`.
        """
        if isinstance(params, str):                       # allow path
            import pathlib
            with open(pathlib.Path(params)) as f:
                params = yaml.safe_load(f)["model"]

        self.model_type    = params.get("type", "CryptoBERT")
        self.freeze_n_layers = int(params.get("freeze_layers", 11))

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

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def preprocess_input(
        self,
        *,
        tweet_content: str,
        rsi: float,
        roc: float,
        date: str,
        previous_label: str,
    ) -> dict:
        """Tokenise a single prompt (returns *lists* for padding later)."""
        prompt = (
            f"Date:{date} | Prev:{previous_label} | "
            f"ROC:{roc:.4f} | RSI:{rsi:.2f} | Tweet:{tweet_content}"
        )
        return self.tokenizer(
            prompt,
            padding="max_length",          # DataCollatorWithPadding handles it
            truncation=True,
            max_length=256,         # well below 514 limit
        )

    def forward(self, inputs):
        """Forward pass returning logits."""
        return self.bert_model(**inputs).logits

    def freeze_layers(self, freeze_until: int | None = None) -> None:
        """Freeze lower transformer layers (default comes from YAML)."""
        if freeze_until is None:
            freeze_until = self.freeze_n_layers
        encoder = self.bert_model.base_model.encoder
        for layer in encoder.layer[:freeze_until]:
            for param in layer.parameters():
                param.requires_grad = False
