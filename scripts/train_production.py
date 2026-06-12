import os
import sys
import time
from spec_bpe.tokenizer import SpecTokenizer

def train_production():
    print("SPEC-BPE 2.0 Production Training Pipeline")
    print("=" * 40)

    vocab_size = 500
    tokenizer = SpecTokenizer(vocab_size=vocab_size)

    # Using local or dummy data if Hub is unreachable
    corpus = """
    Artificial Intelligence is transforming the world.
    Linguistic topology and spectral graph theory provide deep insights into
    how meaning is encoded in subword units. SPEC-BPE uses Density Intelligence
    to create more efficient and stable tokenizations.
    """ * 100

    print(f"Training on {len(corpus)} characters...")
    start_time = time.time()
    tokenizer.train(corpus)
    end_time = time.time() - start_time

    print(f"Training completed in {end_time:.2f}s")
    print(f"Final Vocab Size: {len(tokenizer.vocab)}")
    print(f"Manifold Volume: {tokenizer.manifold_volume:.4f}")

    # Save the production tokenizer
    os.makedirs("models", exist_ok=True)
    model_path = "models/spec_bpe_production.pkl"
    tokenizer.save(model_path)
    print(f"Model saved to {model_path}")

    # Test serialization
    loaded = SpecTokenizer.load(model_path)
    test_text = "The transformer architecture is amazing."
    assert loaded.encode(test_text) == tokenizer.encode(test_text)
    print("Serialization test passed.")

if __name__ == "__main__":
    train_production()
