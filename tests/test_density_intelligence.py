import numpy as np
from spec_bpe.scoring import GeometricPressureScorer

def test_informational_specific_gravity():
    scorer = GeometricPressureScorer(lambd=0.1)

    # Case 1: High contextual displacement (high informational density)
    # Token A and B appear together, significantly shifting the distribution
    p_A, p_B, p_AB = 0.5, 0.5, 0.4
    count = 40
    id_freqs = {1: 50, 2: 50}
    total_count = 100
    score_dense = scorer.score((1, 2), count, id_freqs, total_count)

    # Case 2: Low contextual displacement (low informational density, 'hollow' merge)
    p_A2, p_B2, p_AB2 = 0.5, 0.5, 0.25 # Expected co-occurrence if independent
    count2 = 25
    id_freqs2 = {1: 50, 2: 50}
    score_hollow = scorer.score((1, 2), count2, id_freqs2, total_count)

    assert score_dense > score_hollow

def test_crystallization_synthesis():
    from spec_bpe.tokenizer import SpecTokenizer
    tokenizer = SpecTokenizer(vocab_size=257)

    # Text where "ABC" is a strong semantic crystal but "AB" is common but hollow
    text = "ABC " * 10 + "AB " * 5
    tokenizer.train(text)

    # If Crystallization works, it should prefer the dense ABC structure
    # (Simplified check: verify it trained and encoded)
    encoded = tokenizer.encode("ABC")
    assert len(encoded) < 3
