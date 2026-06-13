# SPEC-BPE (Specificity-guided Pair Encoding) - v2.1

## Introduction
SPEC-BPE shifts the tokenization paradigm from "counting occurrences" to "modeling the topological manifold of language". Version 2.1 introduces enhanced stability, production-ready training pipelines, and integration wrappers.

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from spec_bpe.tokenizer import SpecTokenizer
from spec_bpe.integration import SpecTokenizerTransformerWrapper

# Initialize and train
tokenizer = SpecTokenizer(vocab_size=1000)
tokenizer.train("Your corpus here...")

# Wrap for HF compatibility
wrapper = SpecTokenizerTransformerWrapper(tokenizer)
tokens = wrapper.tokenize("Hello world")
ids = wrapper.encode("Hello world")
```

---

## Production Scripts

The `scripts/` directory contains tools for large-scale application:

- **`train_production.py`**: Trains on HF datasets with support for incremental training and domain-specific normalization (`code_mode`).
  ```bash
  python scripts/train_production.py --dataset "Salesforce/wikitext" --vocab_size 5000
  ```
- **`benchmark.py`**: Compares performance (CR, Fertility) against Tiktoken and GPT-2.
- **`sla_audit.py`**: Verifies SLA-3.0 compliance (Creativity Buffer, Tunneling, Density Intelligence).
- **`assess_languages.py`**: Evaluates typological detection accuracy across diverse languages.

---

## Core Architecture

1. **Non-Commutative Matrix BPE**: Tokens as operators in $G=\text{SL}(2,\mathbb{C})$.
2. **Information-Geometric BPE**: Geodesic paths on the probability manifold.
3. **Algebraic BPE over Finite Fields**: Galois-Hysteresis buffer for linguistic mutations.
4. **Negato-Spectral BPE**: Stochastic boundary tunneling via spectral gap monitoring.

---

## SLA-3.0 recursive Audit Verdict
| Atom | Risk | SLA-3.0 VS | Status |
| :--- | :--- | :--- | :--- |
| **Galois ECC** | Over-denoising | 9.0 | ACTIVE |
| **Negato-Filter** | Morphological Stasis | 8.5 | ACTIVE |
| **Spinor Product** | Path Rigidity | 9.2 | ACTIVE |

---

## Enhancements in v2.1
- **Manifold Stability**: Damped Riemannian expansion with hard volume caps.
- **ACFS**: Asynchronous Computed Field Statistics for global density modeling.
- **Deterministic Scoring**: Fixed-seed Persistent Homology calculations.
- **Domain Tuning**: Customizable normalization for code and structured text.
