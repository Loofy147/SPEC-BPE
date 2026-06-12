import numpy as np
import networkx as nx
from spec_bpe.spectral import SpectralFilter
from spec_bpe.scoring import MatrixScorer, HolonomicMemory

def test_lanczos_efficiency():
    sf = SpectralFilter()
    G = nx.Graph()
    # Create a structured graph to ensure boundaries
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)])
    ids = []
    for u, v in G.edges():
        ids.extend([u, v])
    sf.update_boundaries(ids, 100, force=True)
    assert len(sf.boundaries) > 0
    assert sf.spectral_gap >= 0

def test_lazy_updates():
    sf = SpectralFilter(update_threshold=0.5) # High threshold for test
    # Larger sequence to ensure a valid graph (at least 5 nodes as per spectral.py)
    ids = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    sf.update_boundaries(ids, 100, force=True)
    initial_boundaries = sf.boundaries.copy()
    assert len(initial_boundaries) > 0

    # Add minor change (below threshold)
    ids_minor = ids + [6]
    sf.update_boundaries(ids_minor, 100)
    # Check if lazy update skipped (boundaries should be identical)
    assert sf.boundaries == initial_boundaries

    # Add major change (force update)
    # Adding a different sequence to ensure boundaries actually change
    ids_major = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    sf.update_boundaries(ids_major, 100, force=True)
    # Now it should have changed
    assert sf.boundaries != initial_boundaries

def test_holonomic_stabilization():
    ms = MatrixScorer()
    identity = np.eye(2, dtype=complex)
    ms.holonomic_memory.record_loop(300, identity)
    reward = ms.holonomic_memory.get_reward(300)
    assert reward > 1.0
    ms.confidence[1] = 1.0
    ms.confidence[2] = 1.0
    score_high_conf = ms.score((1, 2))
    ms.confidence[1] = 0.1
    ms.confidence[2] = 0.1
    score_low_conf = ms.score((1, 2))
    assert score_high_conf > score_low_conf

def test_lattice_expansion():
    from spec_bpe.tokenizer import SpecTokenizer
    tokenizer = SpecTokenizer(vocab_size=260)
    initial_volume = tokenizer.manifold_volume
    initial_lambd = tokenizer.geom_scorer.lambd
    tokenizer._riemannian_expansion(10.0)
    assert tokenizer.manifold_volume > initial_volume
    assert tokenizer.geom_scorer.lambd != initial_lambd
