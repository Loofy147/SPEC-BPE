import numpy as np
import pickle
from .utils import get_stats, merge
from .scoring import MatrixScorer, GeometricPressureScorer
from .algebraic import AlgebraicScorer
from .spectral import SpectralFilter

class SpecTokenizer:
    def __init__(self, vocab_size=300, gamma=0.5, lambd=0.05, pi=0.2):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        self.matrix_scorer = MatrixScorer()
        self.geom_scorer = GeometricPressureScorer(lambd=lambd)
        self.alg_scorer = AlgebraicScorer()
        self.spec_filter = SpectralFilter(pi=pi)

        self.manifold_volume = 1.0 # Riemannian Lattice Volume
        self.field_stats = {} # ACFS: Asynchronous Computed Field Statistics

    def save(self, path):
        """Saves the tokenizer state to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Loads a tokenizer state from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _riemannian_expansion(self, xi):
        expansion_rate = np.log(xi + 1.0) / 20.0 # Damped expansion for stability
        self.manifold_volume *= (1.0 + expansion_rate)
        if hasattr(self.spec_filter.ph_scorer, 'current_prior'):
            self.geom_scorer.lambd = self.spec_filter.ph_scorer.current_prior["entropy_target"]
        else:
            self.geom_scorer.lambd /= (1.0 + expansion_rate)

    def _topological_rupture(self, ids):
        distortion_threshold = 5.0
        ruptures = []
        for pair, new_id in list(self.merges.items()):
            s_alg = self.alg_scorer.score(pair)
            if (1.0 / (s_alg + 1e-6)) > distortion_threshold:
                ruptures.append((pair, new_id))

        for pair, new_id in ruptures:
            del self.merges[pair]
            if new_id in self.vocab:
                del self.vocab[new_id]
        return len(ruptures) > 0

    def compute_acfs(self, ids):
        total = len(ids)
        freqs = {}
        for idx in ids:
            freqs[idx] = freqs.get(idx, 0) + 1
        self.field_stats = {idx: count / total for idx, count in freqs.items()}
        return freqs

    def train(self, text):
        if isinstance(text, str):
            text_bytes = text.encode("utf-8")
        else:
            text_bytes = text

        ids = list(text_bytes)

        probe_sample = ids[:100]
        self.spec_filter.update_boundaries(probe_sample, len(self.vocab))
        self.spec_filter.ph_scorer.probe_typology(probe_sample, self.spec_filter.co_occurrence_graph)

        # Train until we reach vocab_size
        while len(self.vocab) < self.vocab_size:
            stats = get_stats(ids)
            if not stats:
                break

            iteration = len(self.vocab) - 256
            if iteration % 5 == 0:
                self.spec_filter.update_boundaries(ids, len(self.vocab), force=(iteration==0))
                xi = self.alg_scorer.get_chaos_index(ids)
                id_freqs = self.compute_acfs(ids)

                if xi > 5.0:
                    self.spec_filter.ph_scorer.probe_typology(ids, self.spec_filter.co_occurrence_graph)
                    self._riemannian_expansion(xi)
                    if self._topological_rupture(ids):
                        stats = get_stats(ids)
                alpha = min(1.0, xi / 10.0)
            else:
                xi = 0.0
                alpha = 0.0
                id_freqs = {idx: count * len(ids) for idx, count in self.field_stats.items()}

            total_count = sum(stats.values())
            best_pair = None
            max_score = -float("inf")

            for pair, count in stats.items():
                if pair[0] not in self.vocab or pair[1] not in self.vocab:
                    continue

                p_B_given_A = count / max(1, id_freqs.get(pair[0], 0))
                p_B = id_freqs.get(pair[1], 0) / (total_count + 1)
                pmi = p_B_given_A / (p_B + 1e-9)

                spec_penalty = self.spec_filter.get_penalty(pair)
                if self.spec_filter.is_forbidden(pair):
                    continue

                s_matrix = self.matrix_scorer.score(pair)
                s_geom = self.geom_scorer.score(pair, count, id_freqs, total_count)
                s_alg = self.alg_scorer.score(pair, xi=xi)

                weighted_pmi = pmi - self.gamma * spec_penalty
                final_score = (alpha * weighted_pmi + (1-alpha) * s_matrix * 10.0) * s_geom * s_alg * self.manifold_volume

                if final_score > max_score:
                    max_score = final_score
                    best_pair = pair

            if best_pair is None or max_score < 0:
                # Fallback: Greedily pick the most frequent valid pair
                valid_pairs = [(p, c) for p, c in stats.items() if p[0] in self.vocab and p[1] in self.vocab and not self.spec_filter.is_forbidden(p)]
                if valid_pairs:
                    best_pair = max(valid_pairs, key=lambda x: x[1])[0]
                else:
                    break

            new_id = 256
            while new_id in self.vocab:
                new_id += 1

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
        changed = True
        while changed:
            changed = False
            stats = get_stats(ids)
            if not stats: break

            best_pair = None
            min_rank = float('inf')
            for pair in stats:
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair

            if best_pair:
                ids = merge(ids, best_pair, self.merges[best_pair])
                changed = True
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        return text_bytes.decode("utf-8", errors="replace")
