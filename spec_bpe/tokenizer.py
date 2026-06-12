import numpy as np
from .utils import get_stats, merge
from .scoring import MatrixScorer, GeometricPressureScorer
from .algebraic import AlgebraicScorer
from .spectral import SpectralFilter

class SpecTokenizer:
    def __init__(self, vocab_size=300, gamma=0.5, lambd=0.1):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        self.matrix_scorer = MatrixScorer()
        self.geom_scorer = GeometricPressureScorer(lambd=lambd)
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

            # Lazy Local Update logic:
            # Re-compute spectral boundaries only when gap indicates transition
            # or periodically.
            if i % 5 == 0:
                self.spec_filter.update_boundaries(ids, len(self.vocab))

            total_count = sum(stats.values())
            id_freqs = {}
            for idx in ids:
                id_freqs[idx] = id_freqs.get(idx, 0) + 1

            best_pair = None
            max_score = -float("inf")

            for pair, count in stats.items():
                p_B_given_A = count / id_freqs[pair[0]]
                p_B = id_freqs[pair[1]] / total_count
                pmi = p_B_given_A / p_B

                penalty = self.gamma if self.spec_filter.is_forbidden(pair) else 0.0

                # S(A->B) = (p(B|A)/p(B)) - Penalty
                base_score = pmi - penalty

                s_matrix = self.matrix_scorer.score(pair)
                s_geom = self.geom_scorer.score(pair, count, id_freqs, total_count)
                s_alg = self.alg_scorer.score(pair)

                # Final SPEC score synthesis
                final_score = base_score * s_matrix * s_geom * s_alg

                if final_score > max_score:
                    max_score = final_score
                    best_pair = pair

            if best_pair is None or max_score < 0:
                break

            new_id = 256 + i
            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

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
            if not stats: break
            # Greedy application of merges
            # Find the first merge that was learned
            candidates = [p for p in stats if p in self.merges]
            if not candidates: break
            pair = min(candidates, key=lambda p: self.merges[p])
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
