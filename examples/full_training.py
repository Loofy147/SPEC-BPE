import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spec_bpe.tokenizer import SpecTokenizer
import time

def run_full_training():
    print("SPEC-BPE 2.0 Full Training Demonstration")
    print("=" * 60)

    # 1. Cold Start: Extremely sparse data (Few-Shot)
    print("\n[Phase 1: Cold Start (Sparse Data)]")
    # A mix of a few technical terms
    cold_start_text = "transformer architecture manifold topology " * 2

    tokenizer = SpecTokenizer(vocab_size=265)
    start_time = time.time()
    tokenizer.train(cold_start_text)

    print(f"Detected Typology: {tokenizer.spec_filter.ph_scorer.current_topology_name}")
    print(f"Manifold Volume: {tokenizer.manifold_volume:.4f}")
    print(f"Warp Factor: {tokenizer.spec_filter.ph_scorer.warp_factor:.4f}")

    encoded = tokenizer.encode("transformer")
    print(f"Encoded 'transformer': {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

    # 2. Topological Rupture & Adaptation: Introduction of chaotic 'noise' or 'slang'
    print("\n[Phase 2: Topological Rupture (New Linguistic Manifold)]")
    # Introduce something very different and chaotic
    chaotic_text = "skibidi gyatt based cringe " * 20

    # Continue training on the new data
    tokenizer.vocab_size = 280
    tokenizer.train(chaotic_text)

    print(f"Post-Chaos Typology: {tokenizer.spec_filter.ph_scorer.current_topology_name}")
    print(f"Manifold Volume (Expanded): {tokenizer.manifold_volume:.4f}")

    # Check if 'transformer' was preserved or ruptured (it might stay if stable)
    encoded_tech = tokenizer.encode("transformer")
    print(f"Encoded 'transformer' post-rupture: {encoded_tech}")

    # Check new units
    encoded_slang = tokenizer.encode("skibidi")
    print(f"Encoded 'skibidi': {encoded_slang}")
    print(f"Decoded: {tokenizer.decode(encoded_slang)}")

    print("\n[Summary]")
    print(f"Total training time: {time.time() - start_time:.2f}s")
    print(f"Final Vocab Size: {len(tokenizer.vocab)}")

if __name__ == "__main__":
    run_full_training()
