from spec_bpe.tokenizer import SpecTokenizer
import time

def run_demo():
    # Demonstrating Morphological and Semantic Coherence
    # We use a text with clear prefix/suffix structure
    text = "undoing redoing redo undoes undoer redoer transformation transformer transform transformed transformative " * 5
    print(f"Input: {text[:80]}...")
    print("-" * 60)

    # We increase vocab size to see more merges
    spec_tokenizer = SpecTokenizer(vocab_size=300, gamma=1.0, lambd=0.01)

    start = time.time()
    spec_tokenizer.train(text)
    train_time = time.time() - start

    encoded = spec_tokenizer.encode(text)
    print(f"SPEC-BPE tokens: {len(encoded)}")
    print(f"Training time: {train_time:.4f}s")

    print("\nLearned Specificity (Subwords > 2 chars):")
    shown = set()
    for pair, idx in spec_tokenizer.merges.items():
        res = spec_tokenizer.vocab[idx]
        if len(res) > 2 and res not in shown:
            try:
                decoded = res.decode('utf-8')
                print(f"  {decoded}")
                shown.add(res)
            except:
                pass
        if len(shown) >= 15:
            break

if __name__ == "__main__":
    run_demo()
