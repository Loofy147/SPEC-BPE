import numpy as np
from .utils import get_stats, merge
from .scoring import MatrixScorer, GeometricScorer
from .algebraic import AlgebraicScorer
from .spectral import SpectralFilter

class SpecTokenizer:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.merges = {} # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}

        self.matrix_scorer = MatrixScorer()
        self.geom_scorer = GeometricScorer()
        self.alg_scorer = AlgebraicScorer()
        self.spec_filter = SpectralFilter()

    def train(self, text):
        if isinstance(text, str):
            text_bytes = text.encode("utf-8")
        else:
            text_bytes = text

        ids = list(text_bytes)
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            # Update spectral boundaries occasionally
            if i % 10 == 0:
                self.spec_filter.update_boundaries(ids, len(self.vocab))

            total_count = sum(stats.values())
            id_freqs = {}
            for idx in ids:
                id_freqs[idx] = id_freqs.get(idx, 0) + 1

            best_pair = None
            max_score = -1.0

            for pair, count in stats.items():
                if self.spec_filter.is_forbidden(pair):
                    continue

                # Directional PMI (Simplified)
                # p(B|A) / p(B) = (count / freq(A)) / (freq(B) / total)
                pmi = (count / id_freqs[pair[0]]) / (id_freqs[pair[1]] / total_count)

                # Composite Score
                s_matrix = self.matrix_scorer.score(pair)
                s_geom = self.geom_scorer.score(pair, count, id_freqs, total_count)
                s_alg = self.alg_scorer.score(pair)

                # Final SPEC score
                score = pmi * s_matrix * s_geom * s_alg

                if score > max_score:
                    max_score = score
                    best_pair = pair

            if best_pair is None:
                break

            new_id = 256 + i
            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Register merge in scorers if needed
            self.matrix_scorer.register_merge(best_pair, new_id)
            self.alg_scorer.register_merge(best_pair, new_id)

    def encode(self, text):
        if isinstance(text, str):
            text_bytes = text.encode("utf-8")
        else:
            text_bytes = text

        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
