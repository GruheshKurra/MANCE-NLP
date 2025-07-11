# MANCE-NLP

**Morphologically Aware Neural Character Embeddings for Natural Language Processing**

A PyTorch implementation of the MANCE method for character-level text processing that achieves better morphological awareness and out-of-vocabulary handling compared to traditional word-level approaches.

## Overview

MANCE (Morphologically Aware Neural Character Embeddings) is a neural approach that processes text at the character level to create word representations that are inherently aware of morphological relationships. This approach offers several advantages over traditional word-level embeddings:

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

## Results

The MANCE method demonstrates:

- Improved handling of morphological variants
- Better out-of-vocabulary word processing
- Reduced vocabulary size requirements
- Competitive or improved accuracy on text classification tasks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{mance2024,
  title={Morphologically Aware Neural Character Embeddings for Natural Language Processing},
  author={Your Name},
  year={2024}
}
```