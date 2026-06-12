import numpy as np

class AlgebraicScorer:
    """
    Algebraic BPE over Finite Fields (F_p)
    Maps tokens to elements of F_251.
    A merge is defined as a "Factorization" or "Minimal Polynomial" logic.
    """
    def __init__(self, p=251):
        self.p = p
        self.token_values = {}

    def _get_val(self, token_id):
        if token_id not in self.token_values:
            # Map to an element in F_p
            # For simplicity, use hash modulo p
            self.token_values[token_id] = hash(token_id) % self.p
        return self.token_values[token_id]

    def score(self, pair):
        v_a = self._get_val(pair[0])
        v_b = self._get_val(pair[1])

        # A merge is "condensed" if it forms a specific algebraic relation.
        # Here we model "common roots" by checking if (v_a + v_b) has a high-degree
        # of symmetry or resides in a specific sub-field (simulated via modulo).

        # Minimal polynomial surrogate:
        # Is (v_a * v_b) + (v_a + v_b) prime in F_p?
        # (Very rough approximation of "irreducible" or "minimal" units)
        val = (v_a * v_b + v_a + v_b) % self.p

        # We prefer "primitive" elements or elements that "factorize" nicely.
        # Score = 1.0 if val is in a "stable" subset of F_p.
        if val in [0, 1, 2, 3, 5, 7, 11, 13]: # Small primes/roots
            return 2.0
        return 1.0

    def register_merge(self, pair, new_id):
        v_a = self._get_val(pair[0])
        v_b = self._get_val(pair[1])
        self.token_values[new_id] = (v_a * v_b) % self.p
