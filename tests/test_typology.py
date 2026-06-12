import numpy as np
import networkx as nx
from spec_bpe.spectral import PersistentHomologyScorer, TypologicalAtlas

def test_fractal_dimension():
    ph_scorer = PersistentHomologyScorer()
    # Simple low-growth sequence (isolating)
    ids_low = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4]
    dim_low = ph_scorer.calculate_fractal_dimension(ids_low)

    # High-growth sequence (agglutinative/complex)
    ids_high = list(range(100))
    dim_high = ph_scorer.calculate_fractal_dimension(ids_high)

    assert dim_high > dim_low

def test_persistence_signature():
    ph_scorer = PersistentHomologyScorer()
    G = nx.Graph()
    # Create a graph with some cycles
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)], weight=1.0)
    sig = ph_scorer.get_persistence_signature(G)
    assert sig > 0.0

def test_atlas_selection():
    ph_scorer = PersistentHomologyScorer()
    atlas = TypologicalAtlas()

    # Test for isolating prior
    name, prior = atlas.get_closest_prior(0.5, 0.1)
    assert name == "isolating"

    # Test for agglutinative prior
    name, prior = atlas.get_closest_prior(0.9, 0.6)
    assert name == "agglutinative"

def test_probe_typology():
    ph_scorer = PersistentHomologyScorer()
    G = nx.Graph()
    G.add_edges_from([(i, i+1) for i in range(10)], weight=1.0)
    ids = list(range(20))

    typology = ph_scorer.probe_typology(ids, G)
    assert typology in ["isolating", "agglutinative", "inflectional"]
    assert ph_scorer.current_topology_name == typology
