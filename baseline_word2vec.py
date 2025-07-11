import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from collections import Counter, defaultdict
import random

class Word2VecEmbedding:
    def __init__(self, vocab_size, embedding_dim=100, window_size=5, num_negative=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_negative = num_negative
        
        # Initialize embeddings
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        nn.init.uniform_(self.in_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.out_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, center_word, context_words, negative_words):
        # Get embeddings
        center_embed = self.in_embeddings(center_word)  # [batch_size, embedding_dim]
        context_embed = self.out_embeddings(context_words)  # [batch_size, embedding_dim]
        negative_embed = self.out_embeddings(negative_words)  # [batch_size, num_negative, embedding_dim]
        
        # Positive score
        pos_score = torch.sum(center_embed * context_embed, dim=1)  # [batch_size]
        pos_loss = -torch.log(torch.sigmoid(pos_score))
        
        # Negative scores
        neg_score = torch.bmm(negative_embed, center_embed.unsqueeze(2)).squeeze(2)  # [batch_size, num_negative]
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1)
        
        return torch.mean(pos_loss + neg_loss)

class TextClassifier(nn.Module):
    def __init__(self, embeddings, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(embeddings.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, text_indices):
        # Average word embeddings for document representation
        embeddings = self.embeddings(text_indices)  # [batch_size, seq_len, embedding_dim]
        doc_embed = torch.mean(embeddings, dim=1)  # [batch_size, embedding_dim]
        
        x = self.relu(self.fc1(doc_embed))
        x = self.dropout(x)
        output = self.fc2(x)
        return output

def preprocess_text(text):
    """Basic text preprocessing"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

def build_vocab(texts, min_count=2):
    """Build vocabulary from texts"""
    word_counts = Counter()
    for text in texts:
        words = preprocess_text(text)
        word_counts.update(words)
    
    # Filter by minimum count
    vocab = {'<UNK>': 0, '<PAD>': 1}
    for word, count in word_counts.items():
        if count >= min_count:
            vocab[word] = len(vocab)
    
    return vocab

def text_to_indices(text, vocab, max_length=100):
    """Convert text to indices"""
    words = preprocess_text(text)
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Pad or truncate
    if len(indices) < max_length:
        indices.extend([vocab['<PAD>']] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    
    return indices

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    # Simple sentiment classification dataset
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
    
    # Create more samples by combining and varying
    texts = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    
    # Add some variation
    for i in range(20):
        if i % 2 == 0:
            texts.append(f"Great movie with excellent story number {i}")
            labels.append(1)
        else:
            texts.append(f"Bad movie with poor story number {i}")
            labels.append(0)
    
    return texts, labels

def train_word2vec_baseline():
    """Train baseline Word2Vec model and classifier"""
    print("Creating sample dataset...")
    texts, labels = create_sample_dataset()
    
    # Build vocabulary
    vocab = build_vocab(texts)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Convert texts to indices
    text_indices = [text_to_indices(text, vocab) for text in texts]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        text_indices, labels, test_size=0.3, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create Word2Vec embeddings (simplified training)
    embedding_dim = 50
    embeddings = nn.Embedding(vocab_size, embedding_dim)
    nn.init.normal_(embeddings.weight, mean=0, std=0.1)
    
    # Create classifier
    classifier = TextClassifier(embeddings, hidden_dim=64, num_classes=2)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    print("Training classifier...")
    for epoch in range(50):
        classifier.train()
        
        # Forward pass
        outputs = classifier(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
    
    # Evaluation
    classifier.eval()
    with torch.no_grad():
        train_outputs = classifier(X_train)
        test_outputs = classifier(X_test)
        
        train_pred = torch.argmax(train_outputs, dim=1)
        test_pred = torch.argmax(test_outputs, dim=1)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\nBaseline Word2Vec Results:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Negative', 'Positive']))
    
    return {
        'model': classifier,
        'vocab': vocab,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'vocab_size': vocab_size
    }

if __name__ == "__main__":
    results = train_word2vec_baseline()
    print(f"Baseline completed with test accuracy: {results['test_acc']:.4f}")