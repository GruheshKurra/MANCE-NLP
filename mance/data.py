import re
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset

PAD, UNK = 0, 1

# Printable ASCII plus German letters: 102 characters, 104 ids with PAD and UNK.
CHARS = [chr(i) for i in range(32, 127)] + list("äöüÄÖÜß")
CHAR_IDS = {c: i + 2 for i, c in enumerate(CHARS)}
NUM_CHARS = len(CHARS) + 2

TOKEN = re.compile(r"\w+|[^\w\s]")

DATASETS = {
    "ag_news": ("fancyzhx/ag_news", None, "cls"),
    "dbpedia": ("fancyzhx/dbpedia_14", None, "cls"),
    # The original GermEval repo only ships a loading script; the parquet branch has the data.
    "germeval14": ("GermanEval/germeval_14", "refs/convert/parquet", "ner"),
}


def tokenize(text):
    # AG News keeps escaped backslashes from the source HTML ("\band").
    return TOKEN.findall(text.replace("\\", " "))


def load_splits(name):
    path, revision, task = DATASETS[name]
    ds = load_dataset(path, revision=revision)
    splits = {}
    for split, rows in ds.items():
        if task == "ner":
            splits[split] = (list(rows["tokens"]), list(rows["ner_tags"]))
        elif name == "dbpedia":
            texts = [f"{t} . {c}" for t, c in zip(rows["title"], rows["content"])]
            splits[split] = ([tokenize(t) for t in texts], list(rows["label"]))
        else:
            splits[split] = ([tokenize(t) for t in rows["text"]], list(rows["label"]))
    label_feature = ds["train"].features["ner_tags" if task == "ner" else "label"]
    names = label_feature.feature.names if task == "ner" else label_feature.names
    return splits, names


def hold_out(splits, fraction, seed):
    """Split a validation set off train. Datasets that ship one are returned unchanged."""
    if "validation" in splits:
        return splits
    tokens, labels = splits["train"]
    order = np.random.default_rng(seed).permutation(len(tokens))
    cut = round(fraction * len(tokens))
    pick = lambda idx: ([tokens[i] for i in idx], [labels[i] for i in idx])
    return {**splits, "validation": pick(order[:cut]), "train": pick(order[cut:])}


class Vocab:
    """Word table for the word-embedding baseline."""

    def __init__(self, sentences, min_freq, max_size):
        counts = Counter(w for s in sentences for w in s)
        self.words = ["<pad>", "<unk>"] + [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.seen = set(counts)

    def __len__(self):
        return len(self.words)


def encode(sentences, labels, task, vocab, max_words, max_chars):
    n = len(sentences)
    chars = np.zeros((n, max_words, max_chars), dtype=np.uint8)
    words = np.zeros((n, max_words), dtype=np.int64)
    for i, sentence in enumerate(sentences):
        for j, w in enumerate(sentence[:max_words]):
            ids = [CHAR_IDS.get(c, UNK) for c in w[:max_chars]]
            chars[i, j, : len(ids)] = ids
            words[i, j] = vocab.index.get(w, UNK)
    if task == "cls":
        y = torch.tensor(labels)
    else:
        y = torch.full((n, max_words), -100)
        for i, tags in enumerate(labels):
            tags = tags[:max_words]
            y[i, : len(tags)] = torch.tensor(tags)
    return {"chars": torch.from_numpy(chars), "words": torch.from_numpy(words), "labels": y}


def dataset_revision(name):
    path, revision, _ = DATASETS[name]
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(path, revision=revision).sha
    except Exception:
        return None
