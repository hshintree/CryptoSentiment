#!/usr/bin/env python3
"""
Quick test script to verify the model forward pass works without device/dtype errors.
"""

import torch
from gpu_scripts.model import Model

def test_model():
    print("Testing Model with prompt-tuning...")
    
    # Initialize model - it will auto-detect and use the best device
    model = Model("config.yaml")
    
    print(f"Model using device: {model.device}")
    print(f"BERT model device: {next(model.bert_model.parameters()).device}")
    print(f"Input embeddings device: {next(model.bert_model.get_input_embeddings().parameters()).device}")
    if hasattr(model, 'prompt_embeddings'):
        print(f"Prompt embeddings device: {model.prompt_embeddings.device}")
    
    # Create sample input
    sample_input = model.preprocess_input(
        tweet_content="Bitcoin is going to the moon!",
        previous_label="Bullish",
        rsi_bucket="neutral",
        roc_bucket="rising"
    )
    
    print("\nSample input keys:", sample_input.keys())
    print("Input IDs shape:", sample_input["input_ids"].shape)
    print("Attention mask shape:", sample_input["attention_mask"].shape)
    
    # Remove token_type_ids since we don't use them
    if "token_type_ids" in sample_input:
        del sample_input["token_type_ids"]
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.bert_model.eval()
    with torch.no_grad():
        try:
            # Model.forward() handles device placement internally
            model_args = {k: v for k, v in sample_input.items() 
                         if k in ["input_ids", "attention_mask"]}
            
            logits = model.forward(model_args)
            print("✅ Forward pass successful!")
            print("Logits shape:", logits.shape)
            print("Logits:", logits)
            
            # Test softmax
            probs = torch.softmax(logits, dim=-1)
            print("Probabilities:", probs)
            
            # Test prediction
            pred = logits.argmax(dim=-1).item()
            pred_label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}[pred]
            print(f"Prediction: {pred_label} (class {pred})")
            
        except Exception as e:
            print("❌ Forward pass failed:")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!") 