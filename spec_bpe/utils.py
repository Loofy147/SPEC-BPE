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
    def __init__(self, preserve_newlines=False, code_mode=False):
        self.preserve_newlines = preserve_newlines
        self.code_mode = code_mode

    def normalize(self, text):
        """Applies Unicode NFKC normalization and optional whitespace handling."""
        text = unicodedata.normalize('NFKC', text)
        if not self.preserve_newlines and not self.code_mode:
            text = re.sub(r'\s+', ' ', text)
        elif self.code_mode:
            # Code mode: compress horizontal spaces but keep structure
            text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    @staticmethod
    def clean(text):
        """Removes non-printable characters while preserving standard whitespace."""
        return "".join(c for c in text if c.isprintable() or c in "\n\r\t")

    def process_corpus(self, corpus, chunk_size=None):
        """Normalizes, cleans, and optionally chunks the corpus."""
        processed = self.clean(self.normalize(corpus))
        if chunk_size:
            return [processed[i:i + chunk_size] for i in range(0, len(processed), chunk_size)]
        return processed

class FastBPE:
    """
    Optimized BPE utilities.
    """
    @staticmethod
    def fast_merge(ids, pair, new_id):
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
