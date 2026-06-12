from spec_bpe.tokenizer import SpecTokenizer
import time

def standard_bpe_mock(text, vocab_size=300):
    from spec_bpe.utils import get_stats, merge
    ids = list(text.encode("utf-8"))
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        if not stats: break
        best = max(stats, key=stats.get)
        new_id = 256 + i
        ids = merge(ids, best, new_id)
        merges[best] = new_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
    return ids, vocab

def run_demo():
    # Longer text for better training
    text = "The transformer transformed the transformation with transformative power. " * 10
    print(f"Original text length: {len(text)}")
    print("-" * 50)

    # Standard BPE
    ids_std, _ = standard_bpe_mock(text, vocab_size=280)
    print(f"Standard BPE token count: {len(ids_std)}")

    # SPEC-BPE
    spec_tokenizer = SpecTokenizer(vocab_size=280)
    start = time.time()
    spec_tokenizer.train(text)
    train_time = time.time() - start

    ids_spec = spec_tokenizer.encode(text)
    print(f"SPEC-BPE token count: {len(ids_spec)}")
    print(f"SPEC-BPE training time: {train_time:.4f}s")

    # Show some learned merges
    print("\nSample learned merges (SPEC-BPE):")
    # Show merges that resulted in actual words/morphemes
    shown = 0
    for pair, idx in spec_tokenizer.merges.items():
        res = spec_tokenizer.vocab[idx]
        if len(res) > 2:
            p0 = spec_tokenizer.vocab[pair[0]]
            p1 = spec_tokenizer.vocab[pair[1]]
            print(f"  {p0} + {p1} -> {res}")
            shown += 1
        if shown >= 10:
            break

if __name__ == "__main__":
    run_demo()
