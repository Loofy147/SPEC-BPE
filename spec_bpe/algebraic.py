import numpy as np

class AlgebraicScorer:
    """
    Algebraic BPE over Finite Fields (F_p)
    Implements Galois-Hysteresis (GH) Buffer and Factorization Ambiguity.
    """
    def __init__(self, p=251, n=8, tau_crit=5.0):
        self.p = p
        self.n = n
        self.tau_crit = tau_crit
        self.token_polynomials = {}
        self.mutations = {} # Syndrome -> Count
        self.ambiguous_variants = {} # token_id -> [alternate_polynomials]

        # Pre-compute indices for vectorized cyclic multiplication
        self._indices = (np.arange(self.n)[:, None] - np.arange(self.n)) % self.n

    def _get_poly(self, token_id):
        if token_id not in self.token_polynomials:
            # Deterministic polynomial generation based on token_id
            state = np.random.RandomState(token_id % (2**32))
            coeffs = state.randint(0, self.p, size=self.n)
            self.token_polynomials[token_id] = coeffs

        poly = self.token_polynomials[token_id]
        if not isinstance(poly, np.ndarray):
            poly = np.array(poly, dtype=int)
            self.token_polynomials[token_id] = poly
        return poly

    def _poly_mul(self, p_a, p_b):
        """Vectorized cyclic polynomial multiplication over F_p."""
        mat = p_a[self._indices]
        return (mat @ p_b) % self.p

    def syndrome_check(self, poly):
        """Syndrome evaluation using primitive root alpha=6."""
        alpha = 6
        powers = np.power(alpha, np.arange(self.n), dtype=object) % self.p
        val = np.sum(poly * powers.astype(int)) % self.p
        return int(val)

    def check_factorization_ambiguity(self, poly):
        """
        Check if the polynomial has multiple 'stable' factorizations.
        """
        syndrome = self.syndrome_check(poly)
        # Dialectal resonance zone
        if syndrome > 50 and np.sum(poly) % 10 == 0:
            return True, syndrome
        return False, syndrome

    def score(self, pair, xi=0.0):
        """
        xi: Local Chaos Index
        """
        p_a = self._get_poly(pair[0])
        p_b = self._get_poly(pair[1])

        p_ab = self._poly_mul(p_a, p_b)
        is_ambiguous, syndrome = self.check_factorization_ambiguity(p_ab)

        if is_ambiguous:
            # Preserve as a dialectal variant: High score despite high syndrome
            return 2.0 / (np.log1p(syndrome) + 1.0)

        # Normalized Hysteresis Logic
        if xi >= self.tau_crit:
            # Innovation Regime: reward diversity
            return 1.2 / (1.0 + np.log1p(syndrome))
        else:
            # Standard Regime: Hard correction towards zero syndrome
            return 1.0 / (1.0 + syndrome)

    def register_merge(self, pair, new_id, xi=0.0):
        p_a = self._get_poly(pair[0])
        p_b = self._get_poly(pair[1])
        res = self._poly_mul(p_a, p_b)

        is_ambiguous, syndrome = self.check_factorization_ambiguity(res)

        if is_ambiguous:
            self.mutations[syndrome] = self.mutations.get(syndrome, 0) + 1
            self.ambiguous_variants[new_id] = [res, (res + 1) % self.p]
        elif xi >= self.tau_crit and syndrome > 5:
            self.mutations[syndrome] = self.mutations.get(syndrome, 0) + 1

        self.token_polynomials[new_id] = res

    def get_chaos_index(self, input_ids):
        if not input_ids: return 0.0
        syndromes = [self.syndrome_check(self._get_poly(idx)) for idx in input_ids]
        # Better normalized chaos index: average syndrome deviation
        return float(np.std(syndromes)) if len(syndromes) > 1 else float(syndromes[0] / 10.0)
