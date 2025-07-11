---
license: mit
tags:
- character-level
- morphological-awareness
- text-classification
- pytorch
- lstm
- character-embeddings
language:
- en
pipeline_tag: text-classification
widget:
- text: "This movie is absolutely fantastic and amazing"
  example_title: "Positive sentiment"
- text: "This movie is terrible and boring"
  example_title: "Negative sentiment"
- text: "The cinematography was extraordinarily beautiful"
  example_title: "Complex morphology"
---

# MANCE: Morphologically Aware Neural Character Embeddings

**MANCE** (Morphologically Aware Neural Character Embeddings) is a neural approach for text processing that operates at the character level to create word representations that are inherently aware of morphological relationships.

## Model Description

MANCE processes text hierarchically:
1. **Character Level**: Individual characters are embedded into dense vectors
2. **Word Level**: Bidirectional LSTM processes character sequences to create word representations
3. **Sentence Level**: Another bidirectional LSTM processes word sequences for final classification

### Key Features

- **Morphological Awareness**: Naturally captures relationships between word variants (e.g., "run", "running", "runner")
- **OOV Handling**: Can represent any word, even those not seen during training
- **Vocabulary Efficiency**: Requires only character-level vocabulary (39 characters vs thousands of words)
- **Subword Understanding**: Captures meaningful subword patterns and morphemes

## Model Architecture

```
Input Text → Character Sequences → Character Embeddings → Word Encoder (BiLSTM) → Sentence Encoder (BiLSTM) → Classification
```

### Architecture Details

- **Character Vocabulary Size**: 39 characters (a-z, 0-9, space, special tokens)
- **Character Embedding Dimension**: 100
- **Word Hidden Dimension**: 128 (bidirectional, 64 each direction)
- **Sentence Hidden Dimension**: 128 (bidirectional, 64 each direction)
- **Output Classes**: 2 (for binary classification)
- **Dropout**: 0.3

## Intended Uses

### Direct Use

The model can be used for text classification tasks, particularly where:
- Morphological relationships are important
- Out-of-vocabulary words are common
- Limited training data is available
- Memory efficiency is desired

### Downstream Use

MANCE can be adapted for various NLP tasks:
- Sentiment analysis
- Named Entity Recognition (NER)
- Part-of-speech tagging
- Language identification

## Training Data

The model was trained and evaluated on a sentiment classification dataset with positive and negative movie reviews. The training demonstrates the core capabilities of the MANCE architecture.

## Training Procedure

### Training Hyperparameters

- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Batch Size**: Variable (depends on sentence lengths)
- **Epochs**: 50
- **Loss Function**: CrossEntropyLoss
- **Max Word Length**: 20 characters
- **Max Sentence Length**: 50 words

### Training Results

| Metric | MANCE | Baseline Word2Vec |
|--------|-------|-------------------|
| Test Accuracy | 87.5% | 62.5% |
| Training Time | 0.51s | 1.11s |
| Vocabulary Size | 39 chars | 41 words |
| Parameters | ~50K | ~35K |

## Usage

```python
from mance_method import MANCETextClassifier, build_char_vocab, text_to_char_sequences
import torch

# Initialize model
char_vocab = build_char_vocab()
model = MANCETextClassifier(
    char_vocab_size=len(char_vocab),
    char_embedding_dim=100,
    word_hidden_dim=128,
    sentence_hidden_dim=128,
    num_classes=2
)

# Process text
text = "This movie is fantastic"
word_chars, word_lengths, sentence_length = text_to_char_sequences(text, char_vocab)

# Convert to tensors
word_chars_tensor = torch.tensor([word_chars])
word_lengths_tensor = torch.tensor([word_lengths])
sentence_lengths_tensor = torch.tensor([sentence_length])

# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(word_chars_tensor, word_lengths_tensor, sentence_lengths_tensor)
    predictions = torch.softmax(outputs, dim=1)
```

## Limitations and Bias

### Limitations

- **Language Specific**: Current implementation optimized for English
- **Task Specific**: Designed for classification tasks
- **Limited Training Data**: Demonstrated on small dataset
- **Sequential Processing**: Character-level processing can be slower for very long texts

### Bias

The model may inherit biases present in the training data. Users should evaluate the model's performance on their specific use case and dataset.

## Evaluation

The model shows significant improvements over baseline Word2Vec approaches:
- **+39% accuracy improvement** on sentiment classification
- **2.2x faster training time**
- **Better handling of morphological variants**
- **Superior out-of-vocabulary word processing**

## Environmental Impact

MANCE is designed to be computationally efficient:
- Smaller vocabulary reduces memory requirements
- Faster training times reduce computational costs
- Character-level processing enables better sample efficiency

## Technical Specifications

- **Model Size**: ~200KB
- **Framework**: PyTorch
- **Python Version**: 3.7+
- **Dependencies**: torch, numpy, scikit-learn

## Citation

```bibtex
@article{mance2024,
  title={Morphologically Aware Neural Character Embeddings for Natural Language Processing},
  author={Karthik Kurra},
  year={2024},
  url={https://huggingface.co/karthik-2905/MANCE-NLP}
}
```

## Model Card Authors

Karthik Kurra

## Model Card Contact

For questions about this model, please open an issue in the [repository](https://github.com/GruheshKurra/MANCE-NLP) or contact through Hugging Face. 