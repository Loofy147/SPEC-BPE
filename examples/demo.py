from spec_bpe.tokenizer import SpecTokenizer
import time

def run_demo():
    # Scenario: Technical Corpus (Hard Regime) vs Chaotic Slang (Social Regime)
    tech_text = "transformation transformer transform transformed transformative " * 10
    slang_text = "skibidi skibidi skibidi skibidi gyatt gyatt gyatt gyatt based based " * 10

    print("SPEC-BPE SLA-3.0 Compliance Demo")
    print("-" * 60)

    # Technical Demo
    print("\n[Regime: Technical (H)]")
    tokenizer_tech = SpecTokenizer(vocab_size=280, pi=0.01) # Low social pressure
    tokenizer_tech.train(tech_text)
    print("Learned Units:")
    shown = set()
    for pair, idx in tokenizer_tech.merges.items():
        res = tokenizer_tech.vocab[idx].decode('utf-8', errors='ignore')
        if len(res) > 3 and res not in shown:
            print(f"  {res}")
            shown.add(res)
        if len(shown) >= 10: break

    # Slang Demo
    print("\n[Regime: Social/Chaotic (S)]")
    tokenizer_slang = SpecTokenizer(vocab_size=280, pi=0.8) # High social pressure
    tokenizer_slang.train(slang_text)
    print("Learned Units (Dynamic Lattice Node Expansion):")
    shown = set()
    for pair, idx in tokenizer_slang.merges.items():
        res = tokenizer_slang.vocab[idx].decode('utf-8', errors='ignore')
        if len(res) > 3 and res not in shown:
            print(f"  {res}")
            shown.add(res)
        if len(shown) >= 10: break

if __name__ == "__main__":
    run_demo()
