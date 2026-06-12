import numpy as np
from spec_bpe.scoring import MatrixScorer, GeometricPressureScorer
from spec_bpe.algebraic import AlgebraicScorer

def test_matrix_scorer():
    scorer = MatrixScorer()
    score = scorer.score((1, 2))
    assert isinstance(score, float)
    assert score > 0

def test_geometric_pressure():
    scorer = GeometricPressureScorer(lambd=0.5)
    id_freqs = {1: 10, 2: 10}
    score = scorer.score((1, 2), 5, id_freqs, 100)
    assert score > 0

def test_algebraic_syndrome():
    scorer = AlgebraicScorer()
    score = scorer.score((1, 2))
    assert score > 0
    assert score <= 1.0
