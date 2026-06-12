import os
import sys
import time
from spec_bpe.tokenizer import SpecTokenizer
from spec_bpe.utils import DataProcessor

def train_production():
    print("SPEC-BPE 2.0 Production Training Pipeline")
    print("=" * 40)

    vocab_size = 1000
    tokenizer = SpecTokenizer(vocab_size=vocab_size)

    # Example corpus
    corpus = """
    Artificial Intelligence is transforming the world.
    Linguistic topology and spectral graph theory provide deep insights into
    how meaning is encoded in subword units. SPEC-BPE uses Density Intelligence
    to create more efficient and stable tokenizations.
    """ * 100

    print(f"Original corpus length: {len(corpus)} characters")
    # Clean and normalize the entire corpus at once to avoid injecting artificial spaces
    processed_corpus = DataProcessor.clean(DataProcessor.normalize(corpus))
    print(f"Normalized corpus length: {len(processed_corpus)} characters")

    print(f"Training on {len(processed_corpus)} characters...")
    start_time = time.time()
    tokenizer.train(processed_corpus)
    end_time = time.time() - start_time

    print(f"Training completed in {end_time:.2f}s")
    print(f"Final Vocab Size: {len(tokenizer.vocab)}")
    print(f"Manifold Volume: {tokenizer.manifold_volume:.4f}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/spec_bpe_production.pkl"
    tokenizer.save(model_path)
    print(f"Model saved to {model_path}")

    loaded = SpecTokenizer.load(model_path)
    test_text = "The transformer architecture is amazing."
    assert loaded.encode(test_text) == tokenizer.encode(test_text)
    print("Serialization test passed.")

if __name__ == "__main__":
    train_production()
