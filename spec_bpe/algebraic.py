import numpy as np

class AlgebraicScorer:
    """
    Algebraic BPE over Finite Fields (F_p)
    Implements Galois-Hysteresis (GH) Buffer and Factorization Ambiguity.
    """
    def __init__(self, p=251, n=8, tau_crit=1.0):
        self.p = p
        self.n = n
        self.tau_crit = tau_crit
        self.token_polynomials = {}
        self.mutations = {} # Syndrome -> Count
        self.ambiguous_variants = {} # token_id -> [alternate_polynomials]

    def _get_poly(self, token_id):
        if token_id not in self.token_polynomials:
            coeffs = np.zeros(self.n, dtype=int)
            coeffs[0] = hash(token_id) % self.p
            coeffs[1] = (hash(token_id) >> 8) % self.p
            self.token_polynomials[token_id] = coeffs
        return self.token_polynomials[token_id]

    def syndrome_check(self, poly):
        alpha = 2
        val = 0
        for i, coeff in enumerate(poly):
            val = (val + coeff * (alpha**i)) % self.p
        return val

    def check_factorization_ambiguity(self, poly):
        """
        Check if the polynomial has multiple 'stable' factorizations.
        In this implementation, we simulate this by checking if the syndrome
        falls into a 'dialectal resonance' zone.
        """
        syndrome = self.syndrome_check(poly)
        # Dialectal resonance: syndrome is high but has high internal symmetry (sum of coeffs is low)
        if syndrome > 50 and np.sum(poly) % 10 == 0:
            return True, syndrome
        return False, syndrome

    def score(self, pair, xi=0.0):
        """
        xi: Local Chaos Index
        """
        p_a = self._get_poly(pair[0])
        p_b = self._get_poly(pair[1])

        # Composition (Polynomial Multiplication)
        p_ab = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                p_ab[(i + j) % self.n] = (p_ab[(i + j) % self.n] + p_a[i] * p_b[j]) % self.p

        is_ambiguous, syndrome = self.check_factorization_ambiguity(p_ab)

        # Structural Consultant Logic
        if is_ambiguous:
            # Preserve as a dialectal variant: High score despite high syndrome
            return 2.0 / (np.log(syndrome + 1) + 1.0)

        # Hysteresis Logic
        if xi >= self.tau_crit:
            # Innovation Regime
            return 1.5 / (syndrome + 0.5)
        else:
            # Standard Regime: Hard correction towards zero syndrome
            return 1.0 / (syndrome + 1.0)

    def register_merge(self, pair, new_id, xi=0.0):
        p_a = self._get_poly(pair[0])
        p_b = self._get_poly(pair[1])
        res = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                res[(i + j) % self.n] = (res[(i + j) % self.n] + p_a[i] * p_b[j]) % self.p

        is_ambiguous, syndrome = self.check_factorization_ambiguity(res)

        if is_ambiguous:
            # Fork the lattice (metaphorically) by registering it as a stable mutation
            self.mutations[syndrome] = self.mutations.get(syndrome, 0) + 1
            # In a full implementation, we would store multiple paths
            self.ambiguous_variants[new_id] = [res, (res + 1) % self.p]
        elif xi >= self.tau_crit and syndrome > 5:
            self.mutations[syndrome] = self.mutations.get(syndrome, 0) + 1

        self.token_polynomials[new_id] = res

    def get_chaos_index(self, input_ids):
        syndromes = [self.syndrome_check(self._get_poly(idx)) for idx in input_ids]
        if not syndromes: return 0.0
        return float(np.std(syndromes))
