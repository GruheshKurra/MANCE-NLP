import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from collections import Counter
import string

class MANCEEmbedding(nn.Module):
    def __init__(self, char_vocab_size, char_embedding_dim=100):
        super().__init__()
        self.char_embedding_dim = char_embedding_dim
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)
        
        # Initialize character embeddings
        nn.init.normal_(self.char_embeddings.weight, mean=0, std=0.1)
    
    def forward(self, char_sequences):
        """
        Args:
            char_sequences: [batch_size, max_word_len] - character indices for each word
        Returns:
            char_embeddings: [batch_size, max_word_len, char_embedding_dim]
        """
        return self.char_embeddings(char_sequences)

class MANCEWordEncoder(nn.Module):
    def __init__(self, char_embedding_dim, word_hidden_dim=128):
        super().__init__()
        self.char_embedding_dim = char_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        
        # Bidirectional LSTM for character-level processing
        self.char_lstm = nn.LSTM(
            char_embedding_dim, 
            word_hidden_dim // 2,  # Divide by 2 for bidirectional
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, char_embeddings, char_lengths):
        """
        Args:
            char_embeddings: [batch_size, max_word_len, char_embedding_dim]
            char_lengths: [batch_size] - actual lengths of each word
        Returns:
            word_representations: [batch_size, word_hidden_dim]
        """
        batch_size, max_word_len, _ = char_embeddings.shape
        
        # Pack sequences for efficient processing
        packed_char_embeddings = nn.utils.rnn.pack_padded_sequence(
            char_embeddings, char_lengths, batch_first=True, enforce_sorted=False
        )
        
        # Process with LSTM
        packed_output, (hidden, _) = self.char_lstm(packed_char_embeddings)
        
        # Extract final hidden states (forward and backward)
        # hidden shape: [2, batch_size, word_hidden_dim // 2]
        forward_hidden = hidden[0]  # Last forward state
        backward_hidden = hidden[1]  # Last backward state
        
        # Concatenate forward and backward hidden states
        word_representation = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return word_representation

class MANCETextClassifier(nn.Module):
    def __init__(self, char_vocab_size, char_embedding_dim=100, word_hidden_dim=128, 
                 sentence_hidden_dim=128, num_classes=2):
        super().__init__()
        
        # Character-level components
        self.char_embeddings = MANCEEmbedding(char_vocab_size, char_embedding_dim)
        self.word_encoder = MANCEWordEncoder(char_embedding_dim, word_hidden_dim)
        
        # Sentence-level LSTM
        self.sentence_lstm = nn.LSTM(
            word_hidden_dim,
            sentence_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification layers
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(sentence_hidden_dim * 2, num_classes)
        
    def forward(self, word_char_sequences, word_lengths, sentence_lengths):
        """
        Args:
            word_char_sequences: [batch_size, max_sentence_len, max_word_len]
            word_lengths: [batch_size, max_sentence_len] - length of each word
            sentence_lengths: [batch_size] - length of each sentence
        """
        batch_size, max_sentence_len, max_word_len = word_char_sequences.shape
        
        # Reshape for word-level processing
        word_chars_flat = word_char_sequences.view(-1, max_word_len)  # [batch_size * max_sentence_len, max_word_len]
        word_lengths_flat = word_lengths.view(-1)  # [batch_size * max_sentence_len]
        
        # Filter out zero-length words (padding)
        valid_words_mask = word_lengths_flat > 0
        valid_word_chars = word_chars_flat[valid_words_mask]
        valid_word_lengths = word_lengths_flat[valid_words_mask]
        
        if len(valid_word_chars) == 0:
            # Handle case with no valid words
            word_representations = torch.zeros(batch_size * max_sentence_len, self.word_encoder.word_hidden_dim)
        else:
            # Get character embeddings for valid words
            char_embeddings = self.char_embeddings(valid_word_chars)
            
            # Encode words
            valid_word_representations = self.word_encoder(char_embeddings, valid_word_lengths)
            
            # Reconstruct full word representations with padding
            word_representations = torch.zeros(
                batch_size * max_sentence_len, 
                self.word_encoder.word_hidden_dim,
                device=valid_word_representations.device
            )
            word_representations[valid_words_mask] = valid_word_representations
        
        # Reshape back to sentence structure
        word_representations = word_representations.view(batch_size, max_sentence_len, -1)
        
        # Pack sequences for sentence-level LSTM
        packed_words = nn.utils.rnn.pack_padded_sequence(
            word_representations, sentence_lengths, batch_first=True, enforce_sorted=False
        )
        
        # Process with sentence-level LSTM
        _, (sentence_hidden, _) = self.sentence_lstm(packed_words)
        
        # Concatenate forward and backward final hidden states
        sentence_representation = torch.cat([sentence_hidden[0], sentence_hidden[1]], dim=1)
        
        # Classification
        output = self.dropout(sentence_representation)
        output = self.classifier(output)
        
        return output

def build_char_vocab():
    """Build character vocabulary"""
    chars = list(string.ascii_lowercase + string.digits + ' ')
    char_vocab = {'<PAD>': 0, '<UNK>': 1}
    for char in chars:
        char_vocab[char] = len(char_vocab)
    return char_vocab

def preprocess_text_mance(text):
    """Preprocess text for MANCE"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def word_to_char_indices(word, char_vocab, max_word_len=20):
    """Convert word to character indices"""
    char_indices = [char_vocab.get(char, char_vocab['<UNK>']) for char in word[:max_word_len]]
    
    # Pad to max_word_len
    while len(char_indices) < max_word_len:
        char_indices.append(char_vocab['<PAD>'])
    
    return char_indices, min(len(word), max_word_len)

def text_to_char_sequences(text, char_vocab, max_sentence_len=50, max_word_len=20):
    """Convert text to character sequences"""
    words = preprocess_text_mance(text)
    
    word_char_sequences = []
    word_lengths = []
    
    for word in words[:max_sentence_len]:
        char_indices, word_len = word_to_char_indices(word, char_vocab, max_word_len)
        word_char_sequences.append(char_indices)
        word_lengths.append(word_len)
    
    # Pad sentence to max_sentence_len
    while len(word_char_sequences) < max_sentence_len:
        word_char_sequences.append([char_vocab['<PAD>']] * max_word_len)
        word_lengths.append(0)
    
    sentence_length = min(len(words), max_sentence_len)
    
    return word_char_sequences, word_lengths, sentence_length

def create_sample_dataset():
    """Create a small sample dataset for testing (same as baseline)"""
    positive_samples = [
        "This movie is absolutely fantastic and amazing",
        "I love this film it's wonderful and great",
        "Excellent acting and brilliant story telling",
        "Best movie ever seen highly recommended",
        "Outstanding performance by all actors",
        "Incredible cinematography and beautiful scenes",
        "Perfect soundtrack and amazing direction",
        "Wonderful characters and engaging plot",
        "Superb writing and excellent execution",
        "Brilliant movie with great entertainment value"
    ]
    
    negative_samples = [
        "This movie is terrible and boring",
        "I hate this film it's awful and bad",
        "Poor acting and horrible story telling",
        "Worst movie ever seen not recommended",
        "Disappointing performance by all actors",
        "Terrible cinematography and ugly scenes",
        "Annoying soundtrack and poor direction",
        "Boring characters and confusing plot",
        "Bad writing and terrible execution",
        "Awful movie with no entertainment value"
    ]
    
    # Create more samples
    texts = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    
    # Add variation
    for i in range(20):
        if i % 2 == 0:
            texts.append(f"Great movie with excellent story number {i}")
            labels.append(1)
        else:
            texts.append(f"Bad movie with poor story number {i}")
            labels.append(0)
    
    return texts, labels

def train_mance_model():
    """Train MANCE model"""
    print("Creating sample dataset...")
    texts, labels = create_sample_dataset()
    
    # Build character vocabulary
    char_vocab = build_char_vocab()
    char_vocab_size = len(char_vocab)
    print(f"Character vocabulary size: {char_vocab_size}")
    
    # Convert texts to character sequences
    max_sentence_len = 30
    max_word_len = 15
    
    print("Converting texts to character sequences...")
    char_data = []
    for text in texts:
        word_chars, word_lens, sent_len = text_to_char_sequences(
            text, char_vocab, max_sentence_len, max_word_len
        )
        char_data.append((word_chars, word_lens, sent_len))
    
    # Prepare data for training
    word_char_sequences = torch.tensor([data[0] for data in char_data])
    word_lengths = torch.tensor([data[1] for data in char_data])
    sentence_lengths = torch.tensor([data[2] for data in char_data])
    labels_tensor = torch.tensor(labels)
    
    # Split data
    indices = torch.randperm(len(texts))
    train_size = int(0.7 * len(texts))
    
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train_chars = word_char_sequences[train_idx]
    X_train_word_lens = word_lengths[train_idx]
    X_train_sent_lens = sentence_lengths[train_idx]
    y_train = labels_tensor[train_idx]
    
    X_test_chars = word_char_sequences[test_idx]
    X_test_word_lens = word_lengths[test_idx]
    X_test_sent_lens = sentence_lengths[test_idx]
    y_test = labels_tensor[test_idx]
    
    print(f"Training samples: {len(X_train_chars)}")
    print(f"Test samples: {len(X_test_chars)}")
    
    # Create MANCE model
    model = MANCETextClassifier(
        char_vocab_size=char_vocab_size,
        char_embedding_dim=50,  # Smaller for demo
        word_hidden_dim=64,
        sentence_hidden_dim=64,
        num_classes=2
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training MANCE model...")
    for epoch in range(50):
        model.train()
        
        # Forward pass
        outputs = model(X_train_chars, X_train_word_lens, X_train_sent_lens)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_chars, X_train_word_lens, X_train_sent_lens)
        test_outputs = model(X_test_chars, X_test_word_lens, X_test_sent_lens)
        
        train_pred = torch.argmax(train_outputs, dim=1)
        test_pred = torch.argmax(test_outputs, dim=1)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\nMANCE Results:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Negative', 'Positive']))
    
    return {
        'model': model,
        'char_vocab': char_vocab,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'char_vocab_size': char_vocab_size
    }

if __name__ == "__main__":
    results = train_mance_model()
    print(f"MANCE completed with test accuracy: {results['test_acc']:.4f}")