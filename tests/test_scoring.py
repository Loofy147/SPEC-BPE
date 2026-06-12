import numpy as np
from spec_bpe.scoring import MatrixScorer, GeometricPressureScorer
from spec_bpe.algebraic import AlgebraicScorer

def test_matrix_scorer():
    scorer = MatrixScorer()
    score = scorer.score((1, 2))
    assert score > 0

def test_geometric_pressure():
    scorer = GeometricPressureScorer(lambd=0.5)
    id_freqs = {1: 10, 2: 10}
    score = scorer.score((1, 2), 5, id_freqs, 100)
    assert score > 0

def test_algebraic_hysteresis():
    scorer = AlgebraicScorer(tau_crit=0.1)
    # Low chaos (xi=0)
    score_low = scorer.score((1, 2), xi=0.0)
    # High chaos (xi=1.0)
    score_high = scorer.score((1, 2), xi=1.0)
    assert score_high >= score_low
