import numpy as np
from spec_bpe.scoring import MatrixScorer, GeometricScorer
from spec_bpe.algebraic import AlgebraicScorer

def test_matrix_scorer():
    scorer = MatrixScorer()
    score = scorer.score((1, 2))
    assert isinstance(score, float)
    assert score >= 0

    scorer.register_merge((1, 2), 256)
    score_new = scorer.score((256, 3))
    assert isinstance(score_new, float)

def test_geometric_scorer():
    scorer = GeometricScorer()
    id_freqs = {1: 10, 2: 10}
    score = scorer.score((1, 2), 5, id_freqs, 100)
    assert isinstance(score, float)
    assert score >= 0

def test_algebraic_scorer():
    scorer = AlgebraicScorer()
    score = scorer.score((1, 2))
    assert score in [1.0, 2.0]
