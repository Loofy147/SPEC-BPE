from spec_bpe.tokenizer import SpecTokenizer

def test_typology_aware_training():
    # Test that training works with new components
    tokenizer = SpecTokenizer(vocab_size=260)
    text = "The quick brown fox jumps over the lazy dog." * 10
    tokenizer.train(text)

    # Check if typology was probed
    assert tokenizer.spec_filter.ph_scorer.current_topology_name in ["isolating", "agglutinative", "inflectional"]

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_sheaf_cohomology_shift():
    # Force a high chaos index to trigger re-probing
    tokenizer = SpecTokenizer(vocab_size=270)
    # Generate very random text to increase xi (variation in syndromes)
    import random
    random_text = "".join(chr(random.randint(0, 255)) for _ in range(1000))
    tokenizer.train(random_text)

    # Since it's random, xi likely spiked
    # We can check if entropy_target was adapted
    current_prior = tokenizer.spec_filter.ph_scorer.current_prior
    assert tokenizer.geom_scorer.lambd == current_prior["entropy_target"]
