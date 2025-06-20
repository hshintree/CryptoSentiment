#!/usr/bin/env python3
"""
Quick test script to verify the model forward pass works without device/dtype errors.
"""

import torch
from model import Model

def test_model():
    print("Testing Model with prompt-tuning...")
    
    # Initialize model
    model = Model("config.yaml")
    
    # Create sample input
    sample_input = model.preprocess_input(
        tweet_content="Bitcoin is going to the moon!",
        rsi=65.0,
        roc=0.025,
        previous_label="Bullish"
    )
    
    print("Sample input keys:", sample_input.keys())
    print("Input IDs shape:", sample_input["input_ids"].shape)
    print("Attention mask shape:", sample_input["attention_mask"].shape)
    
    # Move to device (test on CPU first)
    device = torch.device("cpu")
    model.bert_model.to(device)
    if hasattr(model, 'prompt_embeddings'):
        model.prompt_embeddings.data = model.prompt_embeddings.data.to(device)
    
    batch = {k: v.to(device) for k, v in sample_input.items()}
    
    # Remove position_ids if present (not needed for forward)
    if "position_ids" in batch:
        del batch["position_ids"]
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            model_args = {k: v for k, v in batch.items() 
                         if k in ["input_ids", "attention_mask"]}
            logits = model.forward(model_args)
            print("✅ Forward pass successful!")
            print("Logits shape:", logits.shape)
            print("Logits:", logits)
            
            # Test softmax
            probs = torch.softmax(logits, dim=-1)
            print("Probabilities:", probs)
            
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