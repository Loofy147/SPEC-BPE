from spec_bpe.tokenizer import SpecTokenizer

def test_typology_aware_training():
    tokenizer = SpecTokenizer(vocab_size=260)
    text = "The quick brown fox jumps over the lazy dog." * 10
    tokenizer.train(text)

    assert tokenizer.spec_filter.ph_scorer.current_topology_name in ["isolating", "agglutinative", "inflectional"]

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_sheaf_cohomology_shift():
    tokenizer = SpecTokenizer(vocab_size=270)
    import random
    # High entropy text to trigger chaos
    random_text = "".join(chr(random.randint(0, 255)) for _ in range(1000))
    tokenizer.train(random_text)

    current_prior = tokenizer.spec_filter.ph_scorer.current_prior
    # We check if lambd matches the prior target (set during expansion/shift)
    assert tokenizer.geom_scorer.lambd == current_prior["entropy_target"]
