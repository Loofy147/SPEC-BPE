import numpy as np
from spec_bpe.algebraic import AlgebraicScorer

def test_factorization_ambiguity():
    scorer = AlgebraicScorer()
    # Create a polynomial that triggers ambiguity
    # syndrome > 50 and np.sum(poly) % 10 == 0
    # Let's find one. p=251, alpha=2. syndrome = sum(coeff * 2^i)
    # If we set coeff[0]=60 and others 0. syndrome = 60. sum = 60. 60%10=0.
    poly = np.zeros(8, dtype=int)
    poly[0] = 60

    is_ambiguous, syndrome = scorer.check_factorization_ambiguity(poly)
    assert is_ambiguous
    assert syndrome == 60

    score = scorer.score((1, 2), xi=0.0) # We need to mock _get_poly behavior or use real IDs
    # Mocking _get_poly to return our target poly for a pair
    scorer.token_polynomials[1] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    scorer.token_polynomials[2] = np.array([60, 0, 0, 0, 0, 0, 0, 0])
    # Multiplication of these two: 1*60 = 60 at index 0.

    score = scorer.score((1, 2))
    assert score == 2.0 / (np.log(60 + 1) + 1.0)

def test_register_merge_ambiguity():
    scorer = AlgebraicScorer()
    scorer.token_polynomials[1] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    scorer.token_polynomials[2] = np.array([60, 0, 0, 0, 0, 0, 0, 0])

    scorer.register_merge((1, 2), 300)
    assert 300 in scorer.ambiguous_variants
    assert len(scorer.ambiguous_variants[300]) == 2
