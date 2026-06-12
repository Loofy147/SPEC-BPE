# SPEC-BPE (Specificity-guided Pair Encoding)

## Introduction
The dominant tokenization paradigm in modern Large Language Models (LLMs)—primarily driven by greedy variations of Byte Pair Encoding (BPE), WordPiece, and SentencePiece—operates on a simplistic, frequency-driven heuristic. By iteratively merging the most frequent adjacent character or token pairs, standard BPE treats the lexicon as a static, scalar-indexed set. This methodology induces severe structural pathologies:
- **Morphological Blindness**: It frequently fragments semantically and syntactically coherent units based purely on corpus co-occurrence statistics.
- **Entropy Inequality**: It yields vocabulary distributions where highly frequent, low-information tokens consume significant sequence length, while rare, highly specific tokens suffer from poor representation.
- **Vulnerability to Noise**: Out-of-vocabulary anomalies easily corrupt the vocabulary.

To resolve these limitations, **SPEC-BPE** shifts the tokenization paradigm from "counting occurrences" to "modeling the topological manifold of language".

## 1. Non-Commutative Matrix BPE (The Spinor Tokenizer)
Traditional vector space models represent lexical units as static coordinates. SPEC-BPE models each token $T$ as an operator in a non-commutative matrix space, mapping tokens to elements of a Lie group $G = \text{SL}(2, \mathbb{C})$.

### The Non-Commutative Composition Rule
For a candidate token pair $(A, B)$, composition is defined via the matrix product:
$$M_{AB} = M_A M_B$$
Since $M_A M_B \neq M_B M_A$, the tokenizer naturally distinguishes prefixes from suffixes.

### The Operator-Scoring Function
Merges are evaluated based on the **Spectral Radius** $\rho(M_{AB})$ or the **Trace** of the composed matrix:
$$\text{Score}_{\text{Spinor}}(A, B) = \text{Tr}(M_A M_B)$$

## 2. Information-Geometric BPE (The Geodesic Path)
Under this formulation, every token's contextual probability distribution is equipped with the **Fisher Information Metric (FIM)**.

### The Fisher-Rao Geodesic Distance
For discrete token distributions, the distance resolves to:
$$d_{\text{FR}}(p, q) = 2 \arccos \left( \sum_{i=1}^{V} \sqrt{p_i q_i} \right)$$

### Geodesic Merge Constraining
SPEC-BPE prioritizes merges that represent the shortest path on the probability manifold:
$$\text{Score}_{\text{Geodesic}}(A, B) = \frac{1}{d_{\text{FR}}(p_A, p_{AB}) + d_{\text{FR}}(p_B, p_{AB})}$$

## 3. Algebraic BPE over Finite Fields ($\mathbb{F}_q$)
SPEC-BPE projects character sequences onto a finite field $\mathbb{F}_q$. A sequence of $n$ symbols is mapped to a polynomial $P(x)$:
$$P(x) = \sum_{i=0}^{n-1} t_i x^i$$
A subword merge is represented algebraically as a polynomial factorization. Hierarchical abstraction is leveraged via **Galois field extensions** $\mathbb{F}_{q^d}$.

## 4. Negato-Spectral BPE (Exclusion via Eigenvalues)
Construct a weighted co-occurrence graph and compute the normalized symmetric graph Laplacian $L_{\text{sym}}$.

### The Fiedler Vector
The eigenvector $y_2$ corresponding to the second smallest eigenvalue (algebraic connectivity) provides an optimal bi-partition of the graph.

### Negato-Spectral Filtering
If a candidate merge $(A, B)$ spans a boundary where the Fiedler vector changes sign ($\text{sgn}(y_{2, A}) \neq \text{sgn}(y_{2, B})$), the merge is blocked by a "Negato-filter".

## 5. The Synthesis: SPEC-BPE Architecture
The SPEC-BPE architecture integrates these four paradigms into a reactive, manifold-aware pipeline:
1. **Directional PMI (D-PMI)** with Spectral Penalty:
   $$S(A \to B) = \frac{p(B|A)}{p(B)} - \gamma \cdot \mathbf{1}_{\text{Boundary}}(A, B)$$
2. **Vector Steering**: Gradient utility feedback from the model.
3. **The Lattice Constraint**: Vocabulary as a Riemannian lattice.

---
