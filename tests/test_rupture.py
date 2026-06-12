from spec_bpe.tokenizer import SpecTokenizer

def test_acfs_computation():
    tokenizer = SpecTokenizer()
    ids = [1, 1, 2, 3]
    freqs = tokenizer.compute_acfs(ids)
    assert freqs[1] == 2
    assert tokenizer.field_stats[1] == 0.5

def test_topological_rupture_logic():
    tokenizer = SpecTokenizer(vocab_size=260)
    # Create a dummy merge
    pair = (100, 101)
    new_id = 256
    tokenizer.merges[pair] = new_id
    tokenizer.vocab[new_id] = b"ab"

    # Mock high distortion by manipulating algebraic score
    # distortion = 1 / (s_alg + 1e-6)
    # To get distortion > 5, s_alg must be < 0.2
    # AlgebraicScorer.score returns 1.0 / (syndrome + 1.0)
    # So syndrome must be > 4

    # Let's mock the syndromes of the pair to be high
    poly_high = [100, 100, 100, 100, 100, 100, 100, 100]
    tokenizer.alg_scorer.token_polynomials[100] = poly_high
    tokenizer.alg_scorer.token_polynomials[101] = poly_high

    # Syndrome of p_ab will be very high
    has_rupture = tokenizer._topological_rupture([1, 2, 3])
    assert has_rupture
    assert pair not in tokenizer.merges
    assert new_id not in tokenizer.vocab
