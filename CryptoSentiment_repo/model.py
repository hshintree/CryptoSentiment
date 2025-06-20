"""model.py – Prompt-tuning with frozen BERT
• Uses continuous prompt embeddings prepended to input
• Freezes all BERT parameters except prompt embeddings and classifier head
• Pure text prompts for all features (no numeric_features)
"""

import yaml
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Model:
    """BERT-based model wrapper for financial sentiment analysis."""

    def __init__(self, params: dict | str):
        if isinstance(params, str):
            import pathlib
            with open(pathlib.Path(params)) as f:
                params = yaml.safe_load(f)["model"]

        self.model_type       = params.get("type", "CryptoBERT")
        self.prompt_length    = int(params.get("prompt_length", 10))  # Number of prompt tokens
        self.use_prompt_tuning = params.get("prompt_tuning", True)

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
        
        # Initialize prompt embeddings for prompt-tuning
        if self.use_prompt_tuning:
            hidden_size = self.bert_model.config.hidden_size
            self.prompt_embeddings = nn.Parameter(
                torch.randn(self.prompt_length, hidden_size) * 0.02
            )
            
            # Freeze ALL BERT parameters
            for param in self.bert_model.parameters():
                param.requires_grad = False
                
            # Unfreeze only the classifier head
            for param in self.bert_model.classifier.parameters():
                param.requires_grad = True
                
            print(f"Initialized prompt-tuning with {self.prompt_length} prompt tokens")
            print("Frozen all BERT layers - only training prompt embeddings + classifier")

    def preprocess_input(
        self,
        *,
        tweet_content: str,
        rsi: float,
        roc: float,
        previous_label: str,
        log_volume: float | None = None,
    ) -> dict:
        """
        Tokenise pure text prompt including all numeric features as text.
        No separate numeric_features tensor - everything goes through text.
        """
        # Include volume info in prompt if available
        volume_info = ""
        if log_volume is not None:
            volume_info = f" | Vol(log):{log_volume:.2f}"
            
        # Pure text prompt with all features as readable text
        prompt = (
            f"Prev:{previous_label} | "
            f"ROC:{roc:.4f} | RSI:{rsi:.2f}{volume_info} | Tweet:{tweet_content}"
        )
        
        # Determine safe length
        max_pos = self.bert_model.config.max_position_embeddings
        max_len = min(128, max_pos)  # Reduced from 256 to 128 to be safe

        # Get tokenizer output - only the text prompt
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        
        # No need for manual position_ids - BERT's embedding layer handles this
        # No numeric_features tensor - everything is in the text prompt
        return tokenized

    def forward(self, inputs):
        """Forward pass with prompt embeddings prepended to input."""
        # shortcut to standard fine-tuning
        if not self.use_prompt_tuning:
            return self.bert_model(**inputs).logits

        # 1) get the embeddings for our real tokens
        input_ids      = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        in_embeds      = self.bert_model.get_input_embeddings()(input_ids)

        # 2) prepend our learnable prompt embeddings
        batch_size = in_embeds.size(0)
        prompt_embeds = (
            self.prompt_embeddings.unsqueeze(0)
                                  .expand(batch_size, -1, -1)
                                  .to(in_embeds.device)
        )
        combined_embeds = torch.cat([prompt_embeds, in_embeds], dim=1)

        # 3) build a matching attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.prompt_length,
                                     device=attention_mask.device,
                                     dtype=attention_mask.dtype)
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        # 4) hand it all back to HF — it runs embed→encoder→pooler→classifier
        batch_size = combined_embeds.size(0)
        token_type_ids = torch.zeros(
            batch_size,
            combined_embeds.size(1),
            dtype=torch.long,
            device=combined_embeds.device
        )
        out = self.bert_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            token_type_ids=token_type_ids,
        )
        return out.logits

    def freeze_layers(self, freeze_until: int | None = None) -> None:
        """Freeze lower transformer blocks."""
        if freeze_until is None:
            freeze_until = self._n_freeze_layers
        encoder = self.bert_model.base_model.encoder
        for layer in encoder.layer[:freeze_until]:
            for p in layer.parameters():
                p.requires_grad = False
