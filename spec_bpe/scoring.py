import numpy as np

class HolonomicMemory:
    """
    Identifies sequences that form closed loops in SL(2, C).
    These sequences are 'semantically closed' units (Operational Cohesion).
    """
    def __init__(self, tolerance=1e-3):
        self.loops = {} # token_id -> loop_score
        self.tolerance = tolerance

    def record_loop(self, token_id, matrix):
        identity = np.eye(2, dtype=complex)
        dist = np.linalg.norm(matrix - identity)
        # Crystallization: loops represent stable semantic crystals
        if dist < self.tolerance:
            self.loops[token_id] = 10.0 / (dist + 1e-6)
        else:
            self.loops[token_id] = 1.0 / (dist + 1.0)

    def get_reward(self, token_id):
        return self.loops.get(token_id, 1.0)

class MatrixScorer:
    """
    Non-commutative Matrix BPE (The "Spinor" Tokenizer).
    Models Operational Density and Density Intelligence.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.token_matrices = {}
        self.confidence = {} # token_id -> float
        self.holonomic_memory = HolonomicMemory()

    def _get_matrix(self, token_id):
        if token_id not in self.token_matrices:
            # Generate SL(2, C) matrix: det(M) = 1
            a = self.rng.standard_normal() + 1j * self.rng.standard_normal()
            b = self.rng.standard_normal() + 1j * self.rng.standard_normal()
            c = self.rng.standard_normal() + 1j * self.rng.standard_normal()
            if abs(a) < 1e-5: a = 1.0
            d = (1 + b * c) / a
            self.token_matrices[token_id] = np.array([[a, b], [c, d]], dtype=complex)
            self.confidence[token_id] = 0.5
        return self.token_matrices[token_id]

    def score(self, pair):
        M_A = self._get_matrix(pair[0])
        M_B = self._get_matrix(pair[1])
        M_AB = M_A @ M_B

        identity = np.eye(2, dtype=complex)
        holonomy_dist = np.linalg.norm(M_AB - identity)
        trace = np.trace(M_AB)

        h_reward_A = self.holonomic_memory.get_reward(pair[0])
        h_reward_B = self.holonomic_memory.get_reward(pair[1])

        conf_A = self.confidence.get(pair[0], 0.5)
        conf_B = self.confidence.get(pair[1], 0.5)

        # Operational Specific Gravity
        specific_gravity = float(np.abs(trace)) + 1.0 / (holonomy_dist + 1e-5)

        # Operational Density: reward cohesive operators
        return specific_gravity * (conf_A + conf_B) * (h_reward_A * h_reward_B)

    def register_merge(self, pair, new_id):
        M_A = self._get_matrix(pair[0])
        M_B = self._get_matrix(pair[1])
        M_new = M_A @ M_B
        self.token_matrices[new_id] = M_new
        self.confidence[new_id] = min(1.0, self.confidence.get(pair[0], 0.5) + self.confidence.get(pair[1], 0.5))
        self.holonomic_memory.record_loop(new_id, M_new)

class GeometricPressureScorer:
    """
    Information-Geometric BPE modeling Informational Specific Gravity.
    Prioritizes merges that significantly shift the probability manifold.
    """
    def __init__(self, lambd=0.1, epsilon=1e-10):
        self.lambd = lambd
        self.epsilon = epsilon

    def score(self, pair, pair_freq, id_freqs, total_count):
        p_A = max(self.epsilon, id_freqs.get(pair[0], 0) / total_count)
        p_B = max(self.epsilon, id_freqs.get(pair[1], 0) / total_count)
        p_AB = max(self.epsilon, pair_freq / total_count)

        # Conditional probabilities
        p_B_given_A = pair_freq / max(self.epsilon, id_freqs.get(pair[0], 0))
        p_A_given_B = pair_freq / max(self.epsilon, id_freqs.get(pair[1], 0))

        def d_fr_bernoulli(p, q):
            p = np.clip(p, 0.0, 1.0)
            q = np.clip(q, 0.0, 1.0)
            inner = np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))
            inner = np.clip(inner, -1.0, 1.0)
            return 2.0 * np.arccos(inner)

        # Contextual Displacement: How much does A shift B's distribution?
        dist = d_fr_bernoulli(p_B, p_B_given_A) + d_fr_bernoulli(p_A, p_A_given_B)

        # Informational Specific Gravity: Preference for sharp, dense probability peaks.
        entropy = -p_AB * np.log(p_AB) - (1-p_AB)*np.log(1-p_AB+self.epsilon)

        return (dist + 1.0) / (self.lambd * entropy + self.epsilon)
