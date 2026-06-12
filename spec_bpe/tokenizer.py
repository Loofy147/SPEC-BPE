import numpy as np
from .utils import get_stats, merge
from .scoring import MatrixScorer, GeometricPressureScorer
from .algebraic import AlgebraicScorer
from .spectral import SpectralFilter

class SpecTokenizer:
    def __init__(self, vocab_size=300, gamma=1.0, lambd=0.1, pi=0.1):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        self.matrix_scorer = MatrixScorer()
        self.geom_scorer = GeometricPressureScorer(lambd=lambd)
        self.alg_scorer = AlgebraicScorer()
        self.spec_filter = SpectralFilter(pi=pi)

        self.manifold_volume = 1.0 # Riemannian Lattice Volume

    def _riemannian_expansion(self, xi):
        """
        Dynamically warp the vocabulary space (manifold volume)
        to accommodate validated mutations.
        """
        expansion_rate = np.log(xi + 1.0) / 10.0
        self.manifold_volume *= (1.0 + expansion_rate)
        # Expansion reduces geometric pressure lambd
        # Fix: Using target from prior if available, else scale
        if hasattr(self.spec_filter.ph_scorer, 'current_prior'):
            self.geom_scorer.lambd = self.spec_filter.ph_scorer.current_prior["entropy_target"]
        else:
            self.geom_scorer.lambd /= (1.0 + expansion_rate)

    def train(self, text):
        if isinstance(text, str):
            text_bytes = text.encode("utf-8")
        else:
            text_bytes = text

        ids = list(text_bytes)
        num_merges = self.vocab_size - 256

        probe_sample = ids[:100]
        self.spec_filter.update_boundaries(probe_sample, len(self.vocab))
        self.spec_filter.ph_scorer.probe_typology(probe_sample, self.spec_filter.co_occurrence_graph)

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            if i % 5 == 0:
                self.spec_filter.update_boundaries(ids, len(self.vocab), force=(i==0))
                xi = self.alg_scorer.get_chaos_index(ids)

                # Sheaf-Cohomology Shift
                if xi > 5.0:
                    self.spec_filter.ph_scorer.probe_typology(ids, self.spec_filter.co_occurrence_graph)
                    self._riemannian_expansion(xi)

                alpha = min(1.0, xi / 10.0)
            else:
                xi = 0.0
                alpha = 0.0

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

                spec_penalty = self.spec_filter.get_penalty(pair)
                if self.spec_filter.is_forbidden(pair):
                    continue

                s_matrix = self.matrix_scorer.score(pair)
                s_geom = self.geom_scorer.score(pair, count, id_freqs, total_count)
                s_alg = self.alg_scorer.score(pair, xi=xi)

                # Crystallization: Prioritize density intelligence components
                # pmi is only a 'flat' frequency signal; we weight it by alpha
                weighted_pmi = pmi - self.gamma * spec_penalty

                # Synthesis with manifold volume scaling
                final_score = (alpha * weighted_pmi + (1-alpha) * s_matrix) * s_geom * s_alg * self.manifold_volume

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
            self.alg_scorer.register_merge(best_pair, new_id, xi=xi)

    def encode(self, text):
        if isinstance(text, str):
            text_bytes = text.encode("utf-8")
        else:
            text_bytes = text

        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            if not stats: break
            candidates = [p for p in stats if p in self.merges]
            if not candidates: break
            pair = min(candidates, key=lambda p: self.merges[p])
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
