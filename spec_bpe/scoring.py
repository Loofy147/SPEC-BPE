import numpy as np

class MatrixScorer:
    """
    Non-commutative Matrix BPE (The "Spinor" Tokenizer)
    Maps tokens to SL(2, C) matrices. Merges are matrix products.
    Scoring is based on the Spectral Radius of the resulting matrix.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.token_matrices = {}

    def _get_matrix(self, token_id):
        if token_id not in self.token_matrices:
            # Generate a random SL(2, C) matrix: det(M) = 1
            # M = [[a, b], [c, (1+bc)/a]]
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

        # Spectral radius: max absolute eigenvalue
        try:
            eigenvalues = np.linalg.eigvals(M_AB)
            spectral_radius = np.max(np.abs(eigenvalues))
        except np.linalg.LinAlgError:
            spectral_radius = 1e6 # Penalty for unstable matrices

        # A merge is "stable" if the product matrix resides in a specific "low-energy" region.
        # We return 1 / (1 + spectral_radius) so that lower radius = higher score.
        return 1.0 / (1.0 + float(np.abs(spectral_radius)))

    def register_merge(self, pair, new_id):
        M_A = self._get_matrix(pair[0])
        M_B = self._get_matrix(pair[1])
        self.token_matrices[new_id] = M_A @ M_B

class GeometricScorer:
    """
    Information-Geometric BPE (The Geodesic Path)
    Uses the Fisher Information Metric logic to define distance between distributions.
    Shortest path (least distortion) is prioritized.
    """
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def score(self, pair, pair_freq, id_freqs, total_count):
        # p(A), p(B), p(AB)
        p_A = id_freqs.get(pair[0], 0) / total_count
        p_B = id_freqs.get(pair[1], 0) / total_count
        p_AB = pair_freq / total_count

        # Geodesic distance on the probability simplex (approx via Hellinger or KL)
        # Here we use a simplified Fisher-inspired metric:
        # Distance squared ~ sum (sqrt(p_i) - sqrt(q_i))^2
        # We want to measure the distortion if we replace A, B with AB.

        # Original distribution: [p_A, p_B, rest]
        # New distribution: [0, 0, p_AB, rest]
        # (Assuming AB takes all occurrences of A followed by B)

        dist = np.sqrt(p_A) + np.sqrt(p_B) - np.sqrt(p_AB + self.epsilon)

        # Lower distortion (smaller leap) = higher score
        return 1.0 / (1.0 + float(np.abs(dist)))
