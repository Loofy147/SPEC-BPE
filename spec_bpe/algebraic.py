import numpy as np

class AlgebraicScorer:
    """
    Algebraic BPE over Finite Fields (F_p)
    Implements Galois-Hysteresis (GH) Buffer to handle linguistic mutations.
    """
    def __init__(self, p=251, n=8, tau_crit=1.0):
        self.p = p
        self.n = n
        self.tau_crit = tau_crit
        self.token_polynomials = {}
        self.mutations = {} # Syndrome -> Count

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

    def score(self, pair, xi=0.0):
        """
        xi: Local Chaos Index
        """
        p_a = self._get_poly(pair[0])
        p_b = self._get_poly(pair[1])

        # Composition
        p_ab = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                p_ab[(i + j) % self.n] = (p_ab[(i + j) % self.n] + p_a[i] * p_b[j]) % self.p

        syndrome = self.syndrome_check(p_ab)

        # Hysteresis Logic
        if xi >= self.tau_crit:
            # Innovation Regime: Syndromes are treated as seeds for mutations
            # Lower syndrome might still be preferred, but high syndromes aren't killed.
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

        syndrome = self.syndrome_check(res)
        if xi >= self.tau_crit and syndrome > 5:
            self.mutations[syndrome] = self.mutations.get(syndrome, 0) + 1

        self.token_polynomials[new_id] = res

    def get_chaos_index(self, input_ids):
        # Surrogate for local chaos index: variation in syndrome values
        syndromes = [self.syndrome_check(self._get_poly(idx)) for idx in input_ids]
        if not syndromes: return 0.0
        return float(np.std(syndromes))
