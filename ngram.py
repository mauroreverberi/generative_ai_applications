"""Character n-gram reference model for the company names, without deep learning."""

import math
from collections import Counter

import numpy as np

from tokenizer import END, PAD


class CharNgram:
    """Count-based character model over the last n - 1 characters, with add-k smoothing."""

    def __init__(self, tokenizer, order, k):
        """Store the tokenizer, the order and the smoothing constant."""
        self.tokenizer = tokenizer
        self.order = order
        self.k = k
        self.counts = {}
        self.end_id = tokenizer.token_id_for_char[END]
        self.pad_id = tokenizer.token_id_for_char[PAD]

    def contexts(self, ids):
        """Yield (context, next id) for one name, the context is padded with end markers."""
        padded = [self.end_id] * (self.order - 1) + ids
        for position in range(self.order - 1, len(padded)):
            yield tuple(padded[position - self.order + 1:position]), padded[position]

    def fit(self, name_list):
        """Count every context and next character in the names."""
        for name in name_list:
            for context, next_id in self.contexts(self.tokenizer.encode(name) + [self.end_id]):
                self.counts.setdefault(context, Counter())[next_id] += 1
        return self

    def probabilities(self, context):
        """Return the smoothed probabilities of all tokens except padding after a context."""
        counts = self.counts.get(context, Counter())
        values = np.full(self.tokenizer.vocabulary_size(), self.k)
        values[self.pad_id] = 0.0
        for token_id, count in counts.items():
            values[token_id] += count
        return values / values.sum()

    def loss(self, name_list):
        """Return the mean cross-entropy per character in nats, the end marker included."""
        total, count = 0.0, 0
        for name in name_list:
            for context, next_id in self.contexts(self.tokenizer.encode(name) + [self.end_id]):
                total -= math.log(self.probabilities(context)[next_id])
                count += 1
        return total / count
