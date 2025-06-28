"""model.py – Prompt-tuning with frozen BERT
• Uses continuous prompt embeddings prepended to input
• Freezes all BERT parameters except prompt embeddings and classifier head
• Pure text prompts for all features (no numeric_features)
• Uses weighted ce with class weights, increasing dropout to .3
"""

import os
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

        # enable MPS fallback and pick the right device
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        # load tokenizer + model, then move model to device
        self.tokenizer  = AutoTokenizer.from_pretrained(ckpt)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            ckpt, num_labels=3, hidden_dropout_prob=0.3
        ).to(self.device)
        
        # Eagerly allocate embedding storage on MPS device
        self.bert_model.get_input_embeddings().to(self.device)
        
        # Initialize prompt embeddings for prompt-tuning
        if self.use_prompt_tuning:
            hidden_size = self.bert_model.config.hidden_size
            # create prompt embeddings and eagerly move to device for MPS
            self.prompt_embeddings = nn.Parameter(
                torch.randn(self.prompt_length, hidden_size, device=self.device) * 0.02
            )
            
            # Freeze ALL BERT parameters...
            for param in self.bert_model.parameters():
                param.requires_grad = False

            # ...except the *last* transformer block (per paper)
            for param in self.bert_model.base_model.encoder.layer[-1].parameters():
                param.requires_grad = True
                
            # Unfreeze the classifier head
            for param in self.bert_model.classifier.parameters():
                param.requires_grad = True
                
            print(f"Initialized prompt-tuning with {self.prompt_length} prompt tokens")
            print("Frozen first 11 BERT layers - training prompt embeddings + last layer + classifier")

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
        
        # Tokenize prompt and let HF handle positions & truncation at 128
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return tokenized

    def forward(self, inputs):
        """Forward pass with prompt embeddings prepended to input."""
        # ensure all incoming tensors live on our target device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
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

        # 4) truncate to model max positions and build explicit position ids
        batch_size, seq_len, _ = combined_embeds.size()
        max_pos = self.bert_model.config.max_position_embeddings
        if seq_len > max_pos:
            combined_embeds = combined_embeds[:, :max_pos, :]
            combined_mask   = combined_mask[:, :max_pos] if combined_mask is not None else None
            seq_len = max_pos

        # Create fresh position_ids to avoid HF's internal buffered_token_type_ids mismatches
        position_ids = torch.arange(seq_len, dtype=torch.long, device=combined_embeds.device).unsqueeze(0).expand(batch_size, -1)

        out = self.bert_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            position_ids=position_ids,
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
