import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spec_bpe.tokenizer import SpecTokenizer

def audit_sla_compliance():
    print("SPEC-BPE 2.0 SLA-3.0 Compliance Audit")
    print("=" * 60)

    # 1. Creativity Buffer (Galois-Hysteresis) Audit
    print("\n[Audit 1: Creativity Buffer (Linguistic Mutation)]")
    tokenizer = SpecTokenizer(vocab_size=265)

    # Technical text (low chaos)
    tech_text = "transformer architecture manifold " * 10
    tokenizer.train(tech_text)
    xi_tech = tokenizer.alg_scorer.get_chaos_index(list(tech_text.encode('utf-8')))
    print(f"Tech Text Chaos Index (xi): {xi_tech:.4f}")

    # Chaotic text (high chaos)
    # Introducing 'mutations' that should bypass hard ECC if xi > tau_crit
    chaotic_text = "skibidi gyatt rizzler based " * 20
    tokenizer.vocab_size = 280
    tokenizer.train(chaotic_text)
    xi_chaotic = tokenizer.alg_scorer.get_chaos_index(list(chaotic_text.encode('utf-8')))
    print(f"Chaotic Text Chaos Index (xi): {xi_chaotic:.4f}")

    # Check if mutations were registered
    num_mutations = len(tokenizer.alg_scorer.mutations)
    print(f"Linguistic Mutations Registered: {num_mutations}")

    if num_mutations > 0 or xi_chaotic > tokenizer.alg_scorer.tau_crit:
        print("VERDICT: Creativity Buffer ACTIVE (SLA-3.0 Compliant)")
    else:
        print("VERDICT: Creativity Buffer DORMANT")

    # 2. Stochastic Boundary Tunneling Audit
    print("\n[Audit 2: Stochastic Boundary Tunneling]")
    # We check if penalty is non-binary and affected by Pi
    pair = (104, 101) # 'he'
    tokenizer.spec_filter.boundaries.add(pair)

    tokenizer.spec_filter.pi = 0.001
    penalty_low = tokenizer.spec_filter.get_penalty(pair)

    tokenizer.spec_filter.pi = 1.0
    penalty_high = tokenizer.spec_filter.get_penalty(pair)

    print(f"Penalty (Pi=0.001): {penalty_low:.4f}")
    print(f"Penalty (Pi=1.0): {penalty_high:.4f}")

    if penalty_high < penalty_low:
        print("VERDICT: Stochastic Tunneling ACTIVE (SLA-3.0 Compliant)")
    else:
        print("VERDICT: Stochastic Tunneling FIXED")

    # 3. Informational Specific Gravity Audit
    print("\n[Audit 3: Informational Specific Gravity]")
    # We check if matrix scores are cohesive
    score_identity = tokenizer.matrix_scorer.score((32, 32)) # Space-space
    print(f"Operational Density (Generic): {score_identity:.4f}")

    if score_identity > 0:
        print("VERDICT: Density Intelligence ACTIVE")
    else:
        print("VERDICT: Density Intelligence INACTIVE")

if __name__ == "__main__":
    audit_sla_compliance()
