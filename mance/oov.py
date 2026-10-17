"""Unseen-variant test: replace test words with forms of the same word never seen in training.

Two versions are built:

  rule      inflections of the word's stem that never occur in the training split. Reaches 10%,
            but many are not real words ("teamest", "aftering").
  attested  the same candidates, kept only if they occur at least `min_count` times in lowercase
            in a separate corpus (DBpedia for AG News). These are real words, but only ~1.4% of
            AG News test tokens have one, so the replaced fraction is reported, not forced.

The sampling seed is fixed, so every model is scored on the same perturbed text.
"""
import random
from collections import Counter

INFLECTIONS = ("s", "es", "ed", "ing", "er", "est", "ly")
DERIVATIONS = ("ers", "ness", "ment", "ments", "ion", "ions")
SKIP = set("the and but for with from are was were been this that these those they you his her "
           "their our your not".split())


def stems(word, seen):
    """The word and any stem left by removing one inflection, if that stem is a training word."""
    found = {word}
    for suffix in INFLECTIONS:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            s = word[: -len(suffix)]
            for stem in (s, s + "e", s[:-1] + "y" if s.endswith("i") else "", s[:-1] if s[-1] == s[-2] else ""):
                if len(stem) >= 3 and stem in seen:
                    found.add(stem)
    return found


def attach(stem, suffix):
    if stem.endswith("e") and suffix[0] in "aeiou":
        return stem[:-1] + suffix                                   # make -> making
    if stem.endswith("y") and stem[-2] not in "aeiou" and suffix != "ing":
        return stem[:-1] + ("ies" if suffix in ("s", "es") else "i" + suffix)  # study -> studies
    if (len(stem) <= 4 and suffix[0] in "aeiou" and stem[-1] not in "aeiouwxy"
            and stem[-2] in "aeiou" and stem[-3] not in "aeiou"):
        return stem + stem[-1] + suffix                             # run -> running
    return stem + suffix


def variants(word, seen, derivations=False):
    suffixes = INFLECTIONS + (DERIVATIONS if derivations else ())
    out = set()
    for stem in stems(word, seen):
        out.add(stem)
        out.update(attach(stem, s) for s in suffixes)
    out.discard(word)
    return out


def match_case(source, word):
    if source.isupper() and len(source) > 1:
        return word.upper()
    return word.capitalize() if source[0].isupper() else word


def lowercase_counts(sentences):
    # Lowercase only, so names ("Appling", "Waring") do not pass as ordinary words.
    return Counter(w for s in sentences for w in s if w.isalpha() and w.islower())


def perturb(sentences, train_words, fraction, seed, max_words, lexicon=None, min_count=20):
    seen = {w.lower() for w in train_words}
    options = {}

    def candidates(w):
        if w not in options:
            options[w] = sorted(v for v in variants(w, seen, derivations=lexicon is not None)
                                if v not in seen and (lexicon is None or lexicon[v] >= min_count))
        return options[w]

    total, slots = 0, []
    for i, sentence in enumerate(sentences):
        for j, w in enumerate(sentence[:max_words]):
            total += 1
            lw = w.lower()
            if len(w) >= 4 and w.isalpha() and lw not in SKIP and lw in seen and candidates(lw):
                slots.append((i, j))

    rng = random.Random(seed)
    rng.shuffle(slots)
    chosen = slots[: round(fraction * total)]
    out = [list(s) for s in sentences]
    for i, j in chosen:
        w = out[i][j]
        out[i][j] = match_case(w, rng.choice(candidates(w.lower())))
    stats = {"tokens": total, "eligible": len(slots), "replaced": len(chosen),
             "fraction": len(chosen) / total, "seed": seed}
    return out, stats
