import torch
from torch import nn


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(1) / mask.sum(1).clamp(min=1)


def encoder(dim, heads, ff_dim, layers, dropout):
    layer = nn.TransformerEncoderLayer(dim, heads, ff_dim, dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class CharWordEncoder(nn.Module):
    """Builds a word vector from the ordered sequence of the word's characters.

    pool="transformer" runs a character Transformer and averages its outputs; pool="mean"
    averages the raw character embeddings (the "averaged characters" ablation). With
    positions=False the Transformer sees no character positions ("no character order").
    """

    def __init__(self, num_chars, char_dim, dim, heads, ff_dim, layers, max_chars, dropout,
                 pool="transformer", positions=True):
        super().__init__()
        self.embed = nn.Embedding(num_chars, char_dim, padding_idx=0)
        self.proj = nn.Linear(char_dim, dim)
        self.pool = pool
        self.pos = nn.Embedding(max_chars, dim) if positions else None
        if pool == "transformer":
            self.encoder = encoder(dim, heads, ff_dim, layers, dropout)
            self.norm = nn.LayerNorm(dim)

    def forward(self, chars):
        mask = chars != 0
        x = self.proj(self.embed(chars))
        if self.pool == "mean":
            return masked_mean(x, mask)
        if self.pos is not None:
            x = x + self.pos.weight[: chars.size(1)]
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.norm(masked_mean(x, mask))


class Model(nn.Module):
    """Word vectors -> Transformer over the words -> class (cls) or one tag per word (ner).

    The batch gives each distinct word once (`uniq`) plus an index map (`slots`), so a word
    that appears many times in a batch is encoded once.
    """

    def __init__(self, cfg, num_labels, vocab_size, num_chars):
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        dim = m["dim"]
        self.task = cfg["task"]
        if m["embedding"] == "mance":
            self.words = CharWordEncoder(
                num_chars, m["char_dim"], dim, m["heads"], m["ff_dim"], m["char_layers"],
                d["max_chars"], m["dropout"], pool=m["char_pool"], positions=m["char_positions"])
        else:
            self.words = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.char_input = m["embedding"] == "mance"
        self.pos = nn.Embedding(d["max_words"], dim)
        self.context = encoder(dim, m["heads"], m["ff_dim"], m["word_layers"], m["dropout"])
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(m["dropout"])
        self.head = nn.Linear(dim, num_labels)

    def forward(self, uniq, slots, word_ids, mask):
        x = self.words(uniq)[slots] if self.char_input else self.words(word_ids)
        x = x * mask.unsqueeze(-1) + self.pos.weight[: x.size(1)]
        h = self.norm(self.context(self.dropout(x), src_key_padding_mask=~mask))
        if self.task == "cls":
            h = masked_mean(h, mask)
        return self.head(self.dropout(h))
