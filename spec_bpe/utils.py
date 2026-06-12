import numpy as np
import unicodedata
import re

def get_stats(ids):
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i+1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class DataProcessor:
    """
    Standardizes and prepares text for SPEC-BPE training and inference.
    """
    @staticmethod
    def normalize(text):
        """Applies Unicode NFKC normalization and strips redundant whitespace."""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def clean(text):
        """Removes non-printable characters while preserving standard whitespace."""
        return "".join(c for c in text if c.isprintable() or c in "\n\r\t")

    @classmethod
    def process_corpus(cls, corpus, chunk_size=1000):
        """Normalizes, cleans, and splits a large corpus into manageable chunks."""
        processed = cls.clean(cls.normalize(corpus))
        return [processed[i:i + chunk_size] for i in range(0, len(processed), chunk_size)]
