#!/usr/bin/env python3
"""
Script to create a pre-trained MANCE model for sharing on Hugging Face
"""

import torch
import json
from mance_method import train_mance_model, build_char_vocab

def create_pretrained_model():
    """Create and save a pre-trained MANCE model"""
    
    print("Training MANCE model...")
    
    # Train the model
    results = train_mance_model()
    model = results['model']
    
    # Build character vocabulary
    char_vocab = build_char_vocab()
    
    print(f"Model trained successfully!")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
    print(f"Character Vocabulary Size: {len(char_vocab)}")
    
    # Save the model state dict
    torch.save(model.state_dict(), 'pytorch_model.bin')
    print("Model weights saved to 'pytorch_model.bin'")
    
    # Save the character vocabulary
    with open('char_vocab.json', 'w') as f:
        json.dump(char_vocab, f, indent=2)
    print("Character vocabulary saved to 'char_vocab.json'")
    
    # Create a model info file
    model_info = {
        'model_type': 'MANCE',
        'test_accuracy': results['test_acc'],
        'char_vocab_size': len(char_vocab),
        'architecture': {
            'char_embedding_dim': 100,
            'word_hidden_dim': 128,
            'sentence_hidden_dim': 128,
            'num_classes': 2,
            'max_word_length': 20,
            'max_sentence_length': 50
        },
        'training_info': {
            'epochs': 50,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss'
        }
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("Model info saved to 'model_info.json'")
    
    print("\nFiles created for Hugging Face upload:")
    print("- pytorch_model.bin (model weights)")
    print("- char_vocab.json (character vocabulary)")
    print("- model_info.json (model metadata)")
    print("- config.json (configuration)")
    print("- model_card.md (model documentation)")
    
    return model, char_vocab, results

if __name__ == "__main__":
    create_pretrained_model() 