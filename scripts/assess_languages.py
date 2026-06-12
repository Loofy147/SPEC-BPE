import os
import sys
import numpy as np
from datasets import load_dataset
from spec_bpe.tokenizer import SpecTokenizer

def run_assessment():
    print("Typological Assessment of SPEC-BPE 2.0")
    print("=" * 40)

    # We'll use a small subset of a multilingual dataset
    # Loading 'xnli' or 'common_voice' might be too large, let's try a simple one.
    try:
        # Using 'polyglot_ner' or similar as a proxy for multilingual text if available
        # Alternatively, just use hardcoded samples for speed if dataset load fails
        languages = {
            "English (Analytic/Isolating)": "The quick brown fox jumps over the lazy dog. Complexity is the enemy of execution.",
            "Turkish (Agglutinative)": "Afyonkarahisarlılaştıramadıklarımızdan mısınız? Evdekilere selam söyle.",
            "Finnish (Agglutinative)": "Epäjärjestelmällistyttämättömyydellänsäkäänköhän hän ei saavuttanut mitään.",
        }

        tokenizer = SpecTokenizer(vocab_size=500)

        for lang_name, text in languages.items():
            print(f"\nAssessing {lang_name}...")
            tokenizer.train(text * 5)

            ph_scorer = tokenizer.spec_filter.ph_scorer
            detected_typology = ph_scorer.current_topology_name
            warp_factor = ph_scorer.warp_factor

            print(f"  Detected Typology: {detected_typology}")
            print(f"  Warp Factor: {warp_factor:.4f}")
            print(f"  Manifold Volume: {tokenizer.manifold_volume:.4f}")

            encoded = tokenizer.encode(text)
            print(f"  Encoded Length: {len(encoded)}")

    except Exception as e:
        print(f"Assessment failed: {e}")

if __name__ == "__main__":
    run_assessment()
