#!/usr/bin/env python3
"""
Debug script to understand the model structure and identify the pooler issue.
"""

from transformers import AutoModelForSequenceClassification
import torch

def debug_model_structure():
    print("🔍 Debugging RoBERTa model structure...")
    
    # Load the same model as in our code
    model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)
    
    print("\n📋 Model structure:")
    print(f"Model type: {type(model)}")
    print(f"Base model type: {type(model.base_model)}")
    
    print("\n🔍 Base model components:")
    for name, module in model.base_model.named_children():
        print(f"  {name}: {type(module)}")
    
    print(f"\n❓ Does base_model have pooler? {hasattr(model.base_model, 'pooler')}")
    if hasattr(model.base_model, 'pooler'):
        print(f"   Pooler type: {type(model.base_model.pooler)}")
        print(f"   Pooler value: {model.base_model.pooler}")
    
    print(f"\n🎯 Classifier structure:")
    print(f"  Classifier type: {type(model.classifier)}")
    for name, module in model.classifier.named_children():
        print(f"    {name}: {type(module)}")
    
    # Test forward pass without pooler
    print(f"\n🧪 Testing standard forward pass...")
    sample_input = torch.randint(0, 1000, (1, 10))
    try:
        output = model(input_ids=sample_input)
        print(f"✅ Standard forward pass works: {output.logits.shape}")
    except Exception as e:
        print(f"❌ Standard forward pass failed: {e}")
    
    # Test encoder only
    print(f"\n🧪 Testing encoder-only pass...")
    try:
        embeddings = model.base_model.embeddings(sample_input)
        encoder_output = model.base_model.encoder(embeddings)
        print(f"✅ Encoder output shape: {encoder_output.last_hidden_state.shape}")
        
        # How does the classifier normally get its input?
        print(f"\n🎯 Testing classifier input...")
        # RoBERTa typically uses the first token ([CLS]) for classification
        cls_output = encoder_output.last_hidden_state[:, 0, :]  # First token
        print(f"   CLS token shape: {cls_output.shape}")
        
        classifier_output = model.classifier(cls_output)
        print(f"✅ Classifier output shape: {classifier_output.shape}")
        
    except Exception as e:
        print(f"❌ Manual forward pass failed: {e}")

if __name__ == "__main__":
    debug_model_structure() 