import numpy as np

class MatrixScorer:
    """
    Non-commutative Matrix BPE (The "Spinor" Tokenizer)
    Maps tokens to SL(2, C) matrices. Merges are matrix products.
    Scoring is based on the Trace or Spectral Radius.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.token_matrices = {}

    def _get_matrix(self, token_id):
        if token_id not in self.token_matrices:
            # Generate a random SL(2, C) matrix: det(M) = 1
            a = self.rng.standard_normal() + 1j * self.rng.standard_normal()
            b = self.rng.standard_normal() + 1j * self.rng.standard_normal()
            c = self.rng.standard_normal() + 1j * self.rng.standard_normal()
            if abs(a) < 1e-5: a = 1.0
            d = (1 + b * c) / a
            self.token_matrices[token_id] = np.array([[a, b], [c, d]], dtype=complex)
        return self.token_matrices[token_id]

    def score(self, pair):
        M_A = self._get_matrix(pair[0])
        M_B = self._get_matrix(pair[1])
        M_AB = M_A @ M_B

        # Scoring based on Trace or Spectral Radius
        # Tr(M) = sum(eigenvalues)
        trace = np.trace(M_AB)

        # A merge is "stable" if the resulting operator has a high trace (cohesive)
        return float(np.abs(trace))

    def register_merge(self, pair, new_id):
        M_A = self._get_matrix(pair[0])
        M_B = self._get_matrix(pair[1])
        self.token_matrices[new_id] = M_A @ M_B

class GeometricScorer:
    """
    Information-Geometric BPE (The Geodesic Path)
    Uses the Fisher-Rao Geodesic Distance on the probability manifold.
    d_FR(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))
    """
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def score(self, pair, pair_freq, id_freqs, total_count):
        # We need p_A, p_B, and p_AB (the "joint" distribution)
        # For simplicity, we model the contextual distributions p, q as
        # simplified 2D distributions [p_i, 1-p_i] representing
        # (occurrence, non-occurrence) of the token in the corpus.

        p_A = max(self.epsilon, id_freqs.get(pair[0], 0) / total_count)
        p_B = max(self.epsilon, id_freqs.get(pair[1], 0) / total_count)
        p_AB = max(self.epsilon, pair_freq / total_count)

        # Helper to compute d_FR between two Bernoulli(p) distributions
        def d_fr_bernoulli(p, q):
            # sum(sqrt(p_i * q_i)) = sqrt(p*q) + sqrt((1-p)*(1-q))
            inner = np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))
            inner = np.clip(inner, -1.0, 1.0)
            return 2.0 * np.arccos(inner)

        # Score_Geodesic = 1 / (d_FR(p_A, p_AB) + d_FR(p_B, p_AB))
        dist_A = d_fr_bernoulli(p_A, p_AB)
        dist_B = d_fr_bernoulli(p_B, p_AB)

        return 1.0 / (dist_A + dist_B + self.epsilon)
