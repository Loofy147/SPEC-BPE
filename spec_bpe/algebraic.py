import numpy as np

class AlgebraicScorer:
    """
    Algebraic BPE over Finite Fields (F_p)
    Maps tokens to polynomials P(x) in F_p[x] / (x^n - 1).
    A merge is defined as a "Factorization" or "Minimal Polynomial" logic.
    """
    def __init__(self, p=251, n=8):
        self.p = p
        self.n = n # Depth of subword abstraction
        self.token_polynomials = {}

    def _get_poly(self, token_id):
        if token_id not in self.token_polynomials:
            # Map token to a small polynomial in F_p[x]
            # Represented as an array of coefficients
            coeffs = np.zeros(self.n, dtype=int)
            coeffs[0] = hash(token_id) % self.p
            coeffs[1] = (hash(token_id) >> 8) % self.p
            self.token_polynomials[token_id] = coeffs
        return self.token_polynomials[token_id]

    def score(self, pair):
        p_a = self._get_poly(pair[0])
        p_b = self._get_poly(pair[1])

        # A merge is "condensed" if it forms a common root or minimal polynomial.
        # We simulate this by checking the norm of the polynomial sum in F_p.
        res = (p_a + p_b) % self.p
        norm = np.sum(res**2) % self.p

        # We prefer "primitive" elements (non-zero norm)
        if norm != 0:
            return 2.0
        return 1.0

    def register_merge(self, pair, new_id):
        p_a = self._get_poly(pair[0])
        p_b = self._get_poly(pair[1])

        # Composition in the field: convolution (polynomial multiplication)
        # mod (x^n - 1)
        res = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                res[(i + j) % self.n] = (res[(i + j) % self.n] + p_a[i] * p_b[j]) % self.p
        self.token_polynomials[new_id] = res
