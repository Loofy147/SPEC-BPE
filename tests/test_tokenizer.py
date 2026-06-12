from spec_bpe.tokenizer import SpecTokenizer

def test_train_and_encode():
    tokenizer = SpecTokenizer(vocab_size=260)
    text = "abcabcabc"
    tokenizer.train(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text
    assert len(encoded) < len(text.encode("utf-8"))

def test_spectral_filter():
    # Test that it handles small graphs
    tokenizer = SpecTokenizer(vocab_size=260)
    tokenizer.train("a")
    encoded = tokenizer.encode("a")
    assert tokenizer.decode(encoded) == "a"
