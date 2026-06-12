import numpy as np

class MatrixScorer:
    """
    Non-commutative Matrix BPE (The "Spinor" Tokenizer)
    Maps tokens to SL(2, C) matrices.
    Implements Holonomic Path dependence.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.token_matrices = {}

    def _get_matrix(self, token_id):
        if token_id not in self.token_matrices:
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

        # Holonomic loop check: Does M_AB return to Identity?
        # dist(M_AB, I)
        identity = np.eye(2, dtype=complex)
        holonomy_dist = np.linalg.norm(M_AB - identity)

        # Trace scoring
        trace = np.trace(M_AB)

        # High trace = cohesive. Small holonomy_dist = semantically closed.
        return float(np.abs(trace)) + 1.0 / (holonomy_dist + 1e-5)

    def register_merge(self, pair, new_id):
        M_A = self._get_matrix(pair[0])
        M_B = self._get_matrix(pair[1])
        self.token_matrices[new_id] = M_A @ M_B

class GeometricPressureScorer:
    """
    Information-Geometric BPE with Geometric Pressure Constraint.
    L = d_FR(p, q) + lambda * H(V)
    """
    def __init__(self, lambd=0.1, epsilon=1e-10):
        self.lambd = lambd
        self.epsilon = epsilon

    def score(self, pair, pair_freq, id_freqs, total_count):
        p_A = max(self.epsilon, id_freqs.get(pair[0], 0) / total_count)
        p_B = max(self.epsilon, id_freqs.get(pair[1], 0) / total_count)
        p_AB = max(self.epsilon, pair_freq / total_count)

        def d_fr_bernoulli(p, q):
            inner = np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))
            inner = np.clip(inner, -1.0, 1.0)
            return 2.0 * np.arccos(inner)

        dist = d_fr_bernoulli(p_A, p_AB) + d_fr_bernoulli(p_B, p_AB)

        # Entropy of the pair occurrence (surrogate for H(V) change)
        entropy = -p_AB * np.log(p_AB) - (1-p_AB)*np.log(1-p_AB+self.epsilon)

        # Pressure: Minimize distance but penalize entropy collapse
        # We want small dist and large entropy retention.
        # Score = 1 / (dist + lambda * entropy)
        return 1.0 / (dist + self.lambd * entropy + self.epsilon)
