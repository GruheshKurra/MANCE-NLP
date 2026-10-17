# MANCE: Morphology-Aware Nested Character Embeddings for Word Representation

Code accompanying the paper **"Morphology-Aware Nested Character Embeddings for Word Representation"** (Kurra, Moparthi, Chowdary, Pagadala), IEEE ICCCMLA 2025, published in IEEE Xplore: [document 11580466](https://ieeexplore.ieee.org/document/11580466).

Each word is represented as an ordered sequence of character embeddings drawn from a vocabulary of about 100 characters, so morphological variants (plural, tense, superlative) share most of their representation and unseen words never fall back to an unknown token.

## Paper results

From the published paper (Transformer encoder variant):

| Task | Metric | MANCE |
|---|---|---|
| AG News | Accuracy | 93.5% |
| DBpedia | Accuracy | 99.0% |
| GermEval 2014 (German NER) | F1 | 81.3 |

- **Robustness to unseen variants:** with 10% of test words replaced by morphological variants not seen in training, accuracy dropped by as little as 1.3%, against 7–9% for word-level embeddings.
- **Character order matters:** removing character order cost 1.7 points on AG News (93.5% to 91.8%).

**What this repository contains:** a compact BiLSTM implementation of the MANCE idea with a Word2Vec baseline and a small demonstration comparison. The code for the full AG News, DBpedia and GermEval experiments reported in the paper is not in this repository yet.

## Overview

MANCE is a neural approach that processes text at the character level to create word representations that are inherently aware of morphological relationships. This approach offers several advantages over traditional word-level embeddings:

- **Morphological Awareness**: Naturally captures relationships between word variants (e.g., "run", "running", "runner")
- **OOV Handling**: Can represent any word, even those not seen during training
- **Vocabulary Efficiency**: Requires only character-level vocabulary instead of large word vocabularies
- **Subword Understanding**: Captures meaningful subword patterns and morphemes

## Features

- Character-level embedding layer
- Bidirectional LSTM word encoder
- Sentence-level classification architecture
- Comparison with baseline Word2Vec approach
- Comprehensive evaluation metrics

## Installation

```bash
git clone https://github.com/GruheshKurra/MANCE-NLP.git
cd MANCE-NLP
pip install torch numpy scikit-learn
```

## Usage

### Basic MANCE Model Training

```python
from mance_method import train_mance_model

# Train the MANCE model
results = train_mance_model()
print(f"Test Accuracy: {results['test_acc']:.4f}")
```

### Baseline Comparison

```python
from compare_methods import compare_methods

# Compare MANCE with baseline Word2Vec
comparison_results = compare_methods()
```

### Using MANCE for Text Classification

```python
from mance_method import MANCETextClassifier, build_char_vocab, text_to_char_sequences

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
text = "This is an example sentence"
word_chars, word_lengths, sentence_length = text_to_char_sequences(text, char_vocab)
```

## Model Architecture

The MANCE architecture consists of three main components:

1. **Character Embeddings**: Maps individual characters to dense vectors
2. **Word Encoder**: Bidirectional LSTM that processes character sequences to create word representations
3. **Sentence Encoder**: Processes sequences of word representations for final classification

## File Structure

```
├── mance_method.py          # Main MANCE implementation
├── baseline_word2vec.py     # Baseline Word2Vec implementation
├── compare_methods.py       # Comparison and evaluation scripts
└── README.md               # This file
```

## Demo results

`compare_methods.py` runs a small demonstration on morphological-variant sentences. It is a sanity check of the idea, not a benchmark; the benchmark numbers are in [Paper results](#paper-results).

## Citation

Kurra, G. S. S. K., Moparthi, C., Chowdary, P. M., and Pagadala, P. K. Morphology-Aware Nested Character Embeddings for Word Representation. IEEE ICCCMLA 2025. IEEE Xplore document 11580466.
