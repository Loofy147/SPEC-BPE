import numpy as np
import unicodedata
import re
from collections import defaultdict

def get_stats(ids):
    """
    Returns a dictionary of pair counts.
    """
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i+1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    Returns a new list of ids with the pair replaced by idx.
    """
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

class FastBPE:
    """
    Optimized BPE utilities using linked lists and pair tracking.
    """
    @staticmethod
    def get_stats_with_positions(ids):
        stats = defaultdict(int)
        positions = defaultdict(set)
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            stats[pair] += 1
            positions[pair].add(i)
        return dict(stats), positions

    @staticmethod
    def fast_merge(ids, pair, new_id):
        """
        Standard list-based merge. For very large corpora, a doubly linked list
        would be better, but for our scale, this is efficient enough.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
